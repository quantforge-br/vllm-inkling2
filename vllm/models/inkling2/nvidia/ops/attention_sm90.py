# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative-bias paged attention for Hopper and SM12x (Triton).

Stands in for tml_fa4 (prefill) and the tcgen05 decode/verify kernel of
`ops/attention.py`, reading the same sheared relative logits of
`attention_prologue`: column `c` of token `t` holds key
`c + cdiv(past + tile_end, 128) * 128 - (E + 256)`, where `tile_end` ends
the token's 128-row query tile (see `qr_clip_rel_tokens`).

A program owns `BLOCK_M` rows (tokens x GQA group) of one request and KV
head, and one KV split: split `s` of `S` takes key blocks `s, s + S, ...`
of the rows' visible range, so a split count that depends only on the batch
holds for every replay of a CUDA graph.
"""

import functools
import os

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

# The query tile of tml_fa4 that shears the relative logits.
_SHEAR_TILE = tl.constexpr(128)

# Output of `_rel_attention_kernel`: fp32 output and natural-log LSE (the
# tml_fa4 replacement), the gated bf16 output, or fp32 split partials with
# log2 LSEs (merged by `_rel_decode_combine_kernel`).
MODE_OUT_LSE = 0
MODE_GATED = 1
MODE_PARTIAL = 2


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))


@triton.jit
def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):
    # Load each E4M3 pair once, retaining 16-bit packing for BF16 MMA order.
    half = tl.arange(0, D // 2)
    pair = tl.load(
        ptr.to(tl.pointer_type(tl.uint16))
        + tl.multiple_of(offsets // 2, D // 2)[:, None]
        + half[None, :],
        mask=mask[:, None],
        other=0,
    )
    lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }",
        "=h,=h,h",
        [pair],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    return tl.interleave(lo, hi).to(dtype)


@functools.cache
def _skip_old_bias() -> bool:
    return os.getenv("VLLM_INKLING2_ATTN_SKIP_OLD_BIAS", "0") == "1"


@functools.cache
def prefill_attn_int8() -> bool:
    return os.getenv("VLLM_INKLING2_PREFILL_ATTN_INT8", "0") == "1"


@functools.cache
def prefill_attn_pv_fp16_acc() -> bool:
    return os.getenv("VLLM_INKLING2_PREFILL_ATTN_PV_F16ACC", "0") == "1"


# Smallest query length whose int8 KV prefill takes the int8 QK^T path.
PREFILL_ATTN_INT8_MIN_Q = 64


@functools.cache
def prefill_attn_splits_enabled() -> bool:
    return os.getenv("VLLM_INKLING2_PREFILL_ATTN_SPLITS", "0") == "1"


# Keys a program handles in the time of its fixed per-launch work, and the
# key-equivalents of merging one split's fp32 partials, per query tile.
_SPLIT_FIXED_KEYS = 256
_SPLIT_MERGE_KEYS = 64


def plan_prefill_splits(
    query_lens: np.ndarray, seq_lens: np.ndarray, num_heads: int, num_kv_heads: int
) -> int:
    """KV splits of a full-layer prefill launch
    (`VLLM_INKLING2_PREFILL_ATTN_SPLITS=1`, else 1).

    One 128-row program fills an SM, so a long-context chunk of few query
    tiles leaves its last wave mostly idle. Splitting each tile's keys `s`
    ways trades that tail for merging `s` partials: pick the `s` (up to 8)
    of the smallest estimated span (as `plan_attention`: the larger of the
    wave-quantized longest programs and the total spread over the SMs), in
    units of keys; within 2% of the best resolves to the fewest splits."""
    if not prefill_attn_splits_enabled() or len(query_lens) == 0:
        return 1
    group = num_heads // num_kv_heads
    tokens_per_tile = _block_m(int(query_lens.max()), group) // group
    order = np.argsort(-seq_lens)
    programs = (-(-query_lens[order] // tokens_per_tile) * num_kv_heads).astype(
        np.float64
    )
    splits = np.arange(1, 9, dtype=np.float64)[:, None]
    per_program = seq_lens[order] / splits + _SPLIT_FIXED_KEYS
    waves = np.ceil(np.cumsum(programs * splits, axis=1) / num_compute_units())
    span = np.maximum(
        (waves * per_program).max(axis=1),
        (programs * splits * per_program).sum(axis=1) / num_compute_units(),
    )
    cost = span + np.where(splits[:, 0] > 1, splits[:, 0] * _SPLIT_MERGE_KEYS, 0.0)
    return int(np.argmax(cost <= 1.02 * cost.min())) + 1


@triton.jit
def _relative_scores(
    score,
    pos,
    keys,
    key_valid,
    row_valid,
    rel_row,
    WINDOW: tl.constexpr,
    DRAFT: tl.constexpr,
    E: tl.constexpr,
):
    dist = pos[:, None] - keys[None, :]
    visible = (dist >= 0) & key_valid[None, :]
    if WINDOW > 0:
        visible &= dist < WINDOW
    if not DRAFT:
        in_extent = visible & (dist < E) & row_valid[:, None]
        bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
        score = tl.fma(bias.to(tl.float32), 1.4426950408889634, score)
    return tl.where(visible, score, float("-inf"))


@triton.jit
def _key_mean_kernel(
    K,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Mean,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_bt,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # fp32 mean of the dequantized int8 keys a request appends (one KV head).
    req = tl.program_id(0)
    kv_head = tl.program_id(1)
    q_len = tl.load(CuSeqlensQ + req + 1) - tl.load(CuSeqlensQ + req)
    seqlen_k = tl.load(SeqLens + req)
    offs_d = tl.arange(0, D)
    acc = tl.zeros([D], tl.float32)
    for start in range(seqlen_k - q_len, seqlen_k, BLOCK):
        keys = start + tl.arange(0, BLOCK)
        valid = keys < seqlen_k
        page = tl.load(
            BlockTable + req.to(tl.int64) * stride_bt + keys // PAGE,
            mask=valid,
            other=0,
        )
        ptrs = (
            page.to(tl.int64) * stride_kv_blk
            + (keys % PAGE) * stride_kv_slot
            + kv_head * stride_kv_h
        )
        codes = tl.load(K + ptrs[:, None] + offs_d[None, :], mask=valid[:, None])
        scale = tl.load(
            (K + ptrs + 2 * D).to(tl.pointer_type(tl.float32)), mask=valid, other=0.0
        )
        acc += tl.sum(codes.to(tl.float32) * scale[:, None], 0)
    num_kv_heads = tl.num_programs(1)
    tl.store(
        Mean + (req * num_kv_heads + kv_head) * D + offs_d,
        acc / tl.maximum(q_len, 1).to(tl.float32),
    )


@triton.jit
def _rel_attention_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Out,
    Lse,
    Gate,
    Sink,
    KeyMean,
    softmax_scale,
    num_tokens,
    num_splits,
    stride_q_t,
    stride_q_h,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_rel_t,
    stride_bt,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    DRAFT: tl.constexpr,
    SKIP_OLD_BIAS: tl.constexpr,
    INT8_QK: tl.constexpr,
    PV_F16_ACC: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1) // num_splits
    split = tl.program_id(1) % num_splits
    req = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    first_tok = m_tile * TOKENS
    if first_tok >= q_len:
        return
    seqlen_k = tl.load(SeqLens + req)
    past = seqlen_k - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // G
    row_valid = (rows < TOKENS * G) & (tok < q_len)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(tl.int64)
    if DRAFT:
        pos = tl.zeros([BLOCK_M], tl.int32) + seqlen_k - 1
        pos_lo = seqlen_k - 1
        pos_hi = seqlen_k - 1
    else:
        pos = past + tok
        pos_lo = past + first_tok
        pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1
    # Visible key blocks of the tile, split `split` taking every
    # `num_splits`-th of them.
    n_lo = 0
    if WINDOW > 0:
        n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = tl.minimum(pos_hi + 1, seqlen_k)
    n_hi = (n_hi + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    rel_row = Rel + tl.zeros((BLOCK_M,), tl.int64)
    if not DRAFT:
        tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    if INT8_QK:
        # Symmetric int8 rows of Q: QK^T of the int8 K codes runs on the
        # int8 tensor cores; the row and key scales go on the int32 scores.
        q_f32 = q.to(tl.float32)
        q_amax = tl.max(tl.abs(q_f32), 1)
        q_scale = tl.where(q_amax > 0, q_amax / 127.0, 1.0)
        q_i8 = tl.extra.cuda.libdevice.rint(q_f32 / q_scale[:, None]).to(tl.int8)
        row_scale = q_scale * qk_scale
        # The rounding error of Q shifts every score of a row by about its
        # dot with the mean key (large when keys share a channel offset):
        # softmax ignores the shift, the LSE takes it back.
        key_mean = tl.load(KeyMean + (req * (H // G) + kv_head) * D + offs_d)
        q_err = q_f32 - q_i8.to(tl.float32) * q_scale[:, None]
        lse_shift = tl.sum(q_err * key_mean[None, :], 1) * qk_scale
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h
    for i in range(num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        if K.dtype.element_ty == tl.int8:
            # int8 per-token-head rows `[K(D) | V(D) | K scale | V scale]`
            # (8-byte aligned): exact integer codes in bf16/fp16, each key's
            # fp32 scale applied to its score column.
            kv_ptrs = tl.multiple_of(kv_ptrs, 8)
            k_scale = tl.load(
                (K + kv_ptrs + 2 * D).to(tl.pointer_type(tl.float32)),
                mask=key_valid,
                other=0.0,
            )
            if INT8_QK:
                # Key rows stay K-contiguous, the int8 MMA operand layout.
                k_rows = tl.load(
                    K + kv_ptrs[:, None] + offs_d[None, :],
                    mask=key_valid[:, None],
                    other=0,
                )
                s = tl.dot(q_i8, tl.trans(k_rows), out_dtype=tl.int32).to(tl.float32)
                s = s * row_scale[:, None] * k_scale[None, :]
            else:
                k = tl.load(
                    K + kv_ptrs[None, :] + offs_d[:, None],
                    mask=key_valid[None, :],
                    other=0,
                ).to(q.dtype)
                s = tl.dot(q, k) * (k_scale * qk_scale)[None, :]
        else:
            if K.dtype.element_ty.primitive_bitwidth == 8:
                k = _load_fp8_kv(K, kv_ptrs, key_valid, q.dtype, D).T
            else:
                k = tl.load(
                    K + kv_ptrs[None, :] + offs_d[:, None],
                    mask=key_valid[None, :],
                    other=0.0,
                ).to(q.dtype)
            s = tl.dot(q, k) * qk_scale
        if SKIP_OLD_BIAS:
            # Even the newest key in this block predates the relative extent
            # of the earliest valid query. Keep the split's online sum order.
            last_key = (n + 1) * BLOCK_N - 1
            if last_key >= seqlen_k or last_key > pos_lo - E:
                s = _relative_scores(
                    s, pos, keys, key_valid, row_valid, rel_row, WINDOW, DRAFT, E
                )
        else:
            s = _relative_scores(
                s, pos, keys, key_valid, row_valid, rel_row, WINDOW, DRAFT, E
            )
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if V.dtype.element_ty == tl.int8:
            v = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0,
            ).to(tl.float16 if PV_F16_ACC else q.dtype)
            v_scale = tl.load(
                (V + kv_ptrs + D + 4).to(tl.pointer_type(tl.float32)),
                mask=key_valid,
                other=0.0,
            )
            # Scaling P's columns dequantizes V's rows.
            pv = p * v_scale[None, :]
        else:
            if V.dtype.element_ty.primitive_bitwidth == 8:
                v = _load_fp8_kv(V, kv_ptrs, key_valid, q.dtype, D)
            else:
                v = tl.load(
                    V + kv_ptrs[:, None] + offs_d[None, :],
                    mask=key_valid[:, None],
                    other=0.0,
                ).to(q.dtype)
            pv = p
        if PV_F16_ACC:
            # Per-block fp16 accumulation (2x the fp32-accumulate HMMA rate
            # on GeForce Ampere); the running sum stays fp32.
            pv_blk = tl.dot(pv.to(tl.float16), v, out_dtype=tl.float16)
            acc = acc * alpha[:, None] + pv_blk.to(tl.float32)
        else:
            acc = acc * alpha[:, None] + tl.dot(pv.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    if INT8_QK:
        lse2 += lse_shift
    if MODE == 0:
        tl.store(
            Out
            + token[:, None] * stride_out_t
            + head[:, None] * stride_out_h
            + offs_d[None, :],
            o,
            mask=row_valid[:, None],
        )
        tl.store(
            Lse + head * stride_lse_h + token * stride_lse_t,
            lse2 * 0.6931471805599453,
            mask=row_valid,
        )
    elif MODE == 1:
        cols = head[:, None] * D + offs_d[None, :]
        if not DRAFT:
            gate = tl.load(
                Gate + token[:, None] * stride_gate + cols,
                mask=row_valid[:, None],
                other=0.0,
            )
            sink = tl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
            o = o * _sigmoid(gate.to(tl.float32))
            o = o * _sigmoid(lse2 * 0.6931471805599453 - sink.to(tl.float32))[:, None]
        tl.store(
            Out + token[:, None] * stride_out_t + cols,
            o.to(Out.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * num_tokens + token) * H + head
        tl.store(Lse + part, lse2, mask=row_valid)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


@triton.jit
def _merge_splits_kernel(
    OPart,
    LsePart,
    Out,
    Lse,
    num_tokens,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # `MODE_PARTIAL` split partials -> the `MODE_OUT_LSE` output and LSE.
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    splits = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    part = (splits * num_tokens + token) * H + head
    lse_s = tl.load(LsePart + part, mask=splits < NUM_SPLITS, other=float("-inf"))
    lse_max = tl.max(lse_s)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = tl.exp2(lse_s - lse_max)
    w_sum = tl.sum(w)
    o_s = tl.load(
        OPart + part[:, None] * D + offs_d[None, :],
        mask=(w > 0.0)[:, None],
        other=0.0,
    )
    o = tl.sum(o_s * (w / tl.where(w_sum == 0.0, 1.0, w_sum))[:, None], 0)
    lse = tl.where(
        w_sum == 0.0,
        float("-inf"),
        (tl.log2(tl.where(w_sum == 0.0, 1.0, w_sum)) + lse_max) * 0.6931471805599453,
    )
    tl.store(Out + token * stride_out_t + head * stride_out_h + offs_d, o)
    tl.store(Lse + head * stride_lse_h + token * stride_lse_t, lse)


def _block_m(max_query_len: int, group: int) -> int:
    rows = max_query_len * group
    return 16 if rows <= 16 else 32 if rows <= 32 else 64 if rows <= 64 else 128


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    rel_logits: torch.Tensor | None,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    mode: int,
) -> None:
    """Launch `_rel_attention_kernel` over the requests of `cu_seqlens_q`.

    `rel_logits` (sheared, `(T + 128, H, extent + 256)`) is None for the
    DSpark draft: plain block attention with all queries of a request at
    its last position. `window` (keys up to a query's position, inclusive) is
    0 for full layers. `mode` selects the output: `MODE_OUT_LSE` writes the
    fp32 `out` `(T, H, D)` and natural-log `lse` `(H, T)`; `MODE_GATED`
    the bf16 `(T, H * D)` `out` (gated by `gate` and `sinks` unless
    drafting; a single split); `MODE_PARTIAL` the fp32 `out`
    `(S, T, H, D)` and log2 `lse` `(S, T, H)` partials. `MODE_OUT_LSE`
    with several splits merges their partials into `out` and `lse`.
    """
    num_tokens, num_heads, head_dim = q.shape
    if mode == MODE_OUT_LSE and num_splits > 1:
        assert lse is not None
        o_part = q.new_empty((num_splits, *q.shape), dtype=torch.float32)
        lse_part = q.new_empty((num_splits, num_tokens, num_heads), dtype=torch.float32)
        rel_attention(
            q,
            key_cache,
            value_cache,
            o_part,
            lse_part,
            rel_logits,
            gate,
            sinks,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max_query_len,
            softmax_scale=softmax_scale,
            window=window,
            num_splits=num_splits,
            mode=MODE_PARTIAL,
        )
        _merge_splits_kernel[(num_tokens, num_heads)](
            o_part,
            lse_part,
            out,
            lse,
            num_tokens,
            out.stride(0),
            out.stride(1),
            lse.stride(0),
            lse.stride(1),
            H=num_heads,
            D=head_dim,
            NUM_SPLITS=num_splits,
            BLOCK_S=triton.next_power_of_2(num_splits),
            num_warps=4,
        )
        return
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert q.stride(2) == 1
    assert mode != MODE_GATED or num_splits == 1
    draft = rel_logits is None
    extent = 0
    if rel_logits is not None:
        extent = rel_logits.shape[2] - 256
        assert rel_logits.stride(1) == rel_logits.shape[2]
        assert rel_logits.stride(2) == 1
    if mode == MODE_GATED and not draft:
        assert gate is not None and sinks is not None
        assert gate.stride(1) == sinks.stride(1) == 1
    # Unused pointers keep the launches typed alike.
    rel = q if rel_logits is None else rel_logits
    gate = q if gate is None else gate
    sinks = q if sinks is None else sinks
    if lse is None:
        lse = out
    block_m = _block_m(max_query_len, group)
    tokens_per_tile = block_m // group
    int8_qk = (
        key_cache.dtype == torch.int8
        and mode != MODE_GATED
        and max_query_len >= PREFILL_ATTN_INT8_MIN_Q
        and prefill_attn_int8()
    )
    key_mean = q
    if int8_qk:
        reqs = cache_seqlens.shape[0]
        key_mean = torch.empty(
            (reqs, num_kv_heads, head_dim), dtype=torch.float32, device=q.device
        )
        _key_mean_kernel[(reqs, num_kv_heads)](
            key_cache,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            key_mean,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_table.stride(0),
            D=head_dim,
            PAGE=page,
            BLOCK=64,
        )
    num_reqs = cache_seqlens.shape[0]
    grid = (
        triton.cdiv(max_query_len, tokens_per_tile),
        num_kv_heads * num_splits,
        num_reqs,
    )
    if mode == MODE_OUT_LSE:
        out_strides = (out.stride(0), out.stride(1), lse.stride(0), lse.stride(1))
    elif mode == MODE_GATED:
        assert out.is_contiguous()
        out_strides = (out.stride(0), 0, 0, 0)
    else:
        assert out.is_contiguous() and lse.is_contiguous()
        out_strides = (0, 0, 0, 0)
    _rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        out,
        lse,
        gate,
        sinks,
        key_mean,
        softmax_scale,
        num_tokens,
        num_splits,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        rel.stride(0),
        block_table.stride(0),
        *out_strides,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        G=group,
        D=head_dim,
        E=extent,
        PAGE=page,
        WINDOW=window,
        DRAFT=draft,
        SKIP_OLD_BIAS=(
            not draft
            and window == 0
            and mode != MODE_OUT_LSE
            and max_query_len <= 8
            and key_cache.dtype == torch.int8
            and _skip_old_bias()
        ),
        INT8_QK=int8_qk,
        PV_F16_ACC=int8_qk and prefill_attn_pv_fp16_acc(),
        MODE=mode,
        BLOCK_M=block_m,
        BLOCK_N=64 if block_m >= 128 else 128 if block_m <= 32 else 64,
        num_warps=8 if block_m >= 128 else 4,
        num_stages=2,
    )
