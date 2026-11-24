# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loop message controllers, batched over streams.

A controller blends a fp32 message into a fp32 sublayer input `dest`: from
`h = down(rms([dest, message]))` (bf16), `out = (1 - tanh(forget(h))) * dest +
tanh(gain(h)) * message` in fp32, unfused as the reference. The output feeds
the sublayer's RMSNorm, applied in the same kernel.
"""

import functools
import os

import torch

from vllm.triton_utils import tl, tldevice, triton

from .norm import rsqrt


@triton.jit
def _pair_rms_norm_kernel(
    Dest,
    Message,
    Out,
    stride_dest_s,
    stride_out_s,
    T,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + (s * T + t) * D + cols, mask, other=0)
    dest_rstd = rsqrt(tl.div_rn(tl.sum(dest * dest), D) + eps)
    message_rstd = rsqrt(tl.div_rn(tl.sum(message * message), D) + eps)
    out = Out + s * stride_out_s + t * (2 * D)
    tl.store(out + cols, (dest * dest_rstd).to(Out.dtype.element_ty), mask)
    tl.store(out + D + cols, (message * message_rstd).to(Out.dtype.element_ty), mask)


def pair_rms_norm(
    dest: torch.Tensor, message: torch.Tensor, min_rows: int = 0
) -> torch.Tensor:
    """bf16 `(S, T, 2D)` `[rms(dest), rms(message)]` of fp32 `(S, T, D)`
    inputs, weightless; `dest` may broadcast over streams (stride 0). With
    `min_rows` above T, each stream has that many rows and the extra ones
    are left unset."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    out = message.new_empty(
        (num_streams, max(num_tokens, min_rows), 2 * d), dtype=torch.bfloat16
    )
    if num_tokens:
        _pair_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            out,
            dest.stride(0),
            out.stride(0),
            num_tokens,
            1e-6,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
        )
    return out


@triton.jit
def _blend_rms_norm_kernel(
    Dest,
    Message,
    GainForget,
    NormW,
    Out,
    stride_dest_s,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    row = s * T + t
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + row * D + cols, mask, other=0)
    gain = tldevice.tanh(tl.load(GainForget + row * (2 * D) + cols, mask, other=0))
    forget = tldevice.tanh(
        tl.load(GainForget + row * (2 * D) + D + cols, mask, other=0)
    )
    x = (1.0 - forget) * dest + gain * message
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), D) + norm_eps)
    y = x * rstd
    y = y * tl.load(NormW + s * stride_w_s + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * D + cols, y.to(Out.dtype.element_ty), mask)


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """bf16 `(S, T, D)` RMSNorm (row `s` of `norm_weight` `(S, D)`) of
    `(1 - tanh(forget)) * dest + tanh(gain) * message`, with fp32 `(S, T, D)`
    `dest` (may broadcast over streams) and `message`, and fp32 `(S, T, 2D)`
    `gain_forget` = `[gain, forget]`. `norm_weight` may broadcast too."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert gain_forget.shape == (num_streams, num_tokens, 2 * d)
    assert gain_forget.is_contiguous() and gain_forget.dtype == torch.float32
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    if num_tokens:
        _blend_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            gain_forget,
            norm_weight,
            out,
            dest.stride(0),
            norm_weight.stride(0),
            num_tokens,
            norm_eps,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
            # The blend is unfused (no FMA), as the reference.
            enable_fp_fusion=False,
        )
    return out


# Up to this many tokens, three wide launches run a whole controller (from
# 16 tokens the per-stream GEMMs are faster on RTX 3090).
CONTROLLER_MAX_TOKENS = 8


@functools.cache
def _fused_down_norm() -> bool:
    """Whether `controller_blend` normalizes inside the down projection
    (`VLLM_INKLING2_FUSED_CONTROLLER_NORM`, default on)."""
    return os.environ.get("VLLM_INKLING2_FUSED_CONTROLLER_NORM", "1") == "1"


# Input columns per `_controller_down_kernel` program.
_DOWN_SLICE = 256
# Output columns per `_controller_blend_kernel` program.
_BLEND_SLICE = 256
# Tiles and warps from a sweep on RTX 3090 (1 to 8 tokens, 1 and 3 streams).
_DOWN_WARPS = 4
_BLEND_WARPS = 8


@triton.jit
def _controller_down_kernel(
    Pair,
    Down,
    Partials,
    stride_pair_s,
    T,
    D2: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
):
    """fp32 partial `[R]` of `down @ pair` over one `BK`-column slice of one
    (stream, token) row; the slices are summed in order by the blend."""
    g = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    s = tl.program_id(2).to(tl.int64)
    ks = g * BK + tl.arange(0, BK)
    ranks = tl.arange(0, R)
    first = tl.arange(0, 16) == 0
    x = tl.load(Pair + s * stride_pair_s + t * D2 + ks)
    x = tl.where(first[None, :], x[:, None], 0.0).to(x.dtype)
    w = tl.load(Down + s * (R * D2) + ranks[:, None] * D2 + ks[None, :])
    acc = tl.dot(w, x)
    partial = tl.sum(tl.where(first[None, :], acc, 0.0), 1)
    tl.store(Partials + ((s * T + t) * (D2 // BK) + g) * R + ranks, partial)


@triton.jit
def _controller_down_norm_kernel(
    Dest,
    Message,
    Down,
    Partials,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """`_controller_down_kernel` on the unnormalized fp32 rows: each program
    RMS-normalizes its slice itself (as `_pair_rms_norm_kernel`), so the
    `(S, T, 2D)` pair never goes through memory."""
    g = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    s = tl.program_id(2).to(tl.int64)
    half: tl.constexpr = D // BK
    if g < half:  # noqa: SIM108 - constexpr branch avoids evaluating both pointers
        src = Dest + s * stride_dest_s + t * D
    else:
        src = Message + (s * T + t) * D
    cols = tl.arange(0, BLOCK)
    row = tl.load(src + cols, cols < D, other=0)
    rstd = rsqrt(tl.div_rn(tl.sum(row * row), D) + eps)
    x = (tl.load(src + (g % half) * BK + tl.arange(0, BK)) * rstd).to(tl.bfloat16)
    ks = g * BK + tl.arange(0, BK)
    ranks = tl.arange(0, R)
    first = tl.arange(0, 16) == 0
    x = tl.where(first[None, :], x[:, None], 0.0).to(x.dtype)
    w = tl.load(Down + s * (R * 2 * D) + ranks[:, None] * (2 * D) + ks[None, :])
    acc = tl.dot(w, x)
    partial = tl.sum(tl.where(first[None, :], acc, 0.0), 1)
    tl.store(Partials + ((s * T + t) * (2 * D // BK) + g) * R + ranks, partial)


@triton.jit
def _controller_blend_kernel(
    Dest,
    Message,
    Partials,
    GainForget,
    Mixed,
    SumSq,
    stride_dest_s,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
    G_POW2: tl.constexpr,
    BN: tl.constexpr,
):
    """The fp32 blend of one `BN`-column slice of one (stream, token) row
    into `Mixed`, and the slice's sum of squares into `SumSq`."""
    c = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    s = tl.program_id(2).to(tl.int64)
    row = s * T + t
    ranks = tl.arange(0, R)
    slices = tl.arange(0, G_POW2)
    first = tl.arange(0, 16) == 0
    partials = tl.load(
        Partials + (row * G + slices[:, None]) * R + ranks[None, :],
        slices[:, None] < G,
        other=0.0,
    )
    # Summed in fp32, then rounded once to bf16 (as the reference's GEMM).
    hidden = tl.sum(partials, 0)
    hidden = tl.where(first[None, :], hidden[:, None], 0.0).to(tl.bfloat16)
    cols = c * BN + tl.arange(0, BN)
    # Gain rows, then the forget rows of the same columns, in one product.
    both = tl.arange(0, 2 * BN)
    rows = c * BN + both % BN + tl.where(both >= BN, D, 0)
    weights = GainForget + s * (2 * D * R) + rows[:, None] * R + ranks[None, :]
    product = tl.dot(tl.load(weights), hidden)
    product = tl.sum(tl.where(first[None, :], product, 0.0), 1)
    gain, forget = tl.split(tl.permute(tl.reshape(product, (2, BN)), (1, 0)))
    gain = tldevice.tanh(gain)
    forget = tldevice.tanh(forget)
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols)
    message = tl.load(Message + row * D + cols)
    x = (1.0 - forget) * dest + gain * message
    tl.store(Mixed + row * D + cols, x)
    tl.store(
        SumSq + row * (D // BN) + c + tl.arange(0, 1),
        tl.sum(x * x) + tl.zeros((1,), tl.float32),
    )


@functools.cache
def _grouped_controller() -> tuple[int, bool]:
    group = int(os.getenv("VLLM_INKLING2_CONTROLLER_TOKEN_GROUP", "1"))
    if group not in (1, 2, 4, 8):
        raise ValueError("Inkling 2 controller token group must be 1, 2, 4 or 8")
    return group, os.getenv("VLLM_INKLING2_CONTROLLER_RSTD_PREPASS", "0") == "1"


@triton.jit
def _controller_rstd(src, row_ok, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    row = tl.load(src + cols, row_ok & (cols < D), other=0)
    return rsqrt(tl.div_rn(tl.sum(row * row), D) + eps)


@triton.jit
def _controller_rstd_kernel(
    Dest,
    Message,
    Rstd,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    half = tl.program_id(2)
    row = s * T + t
    src = tl.where(half == 0, Dest + s * stride_dest_s + t * D, Message + row * D)
    rstd = _controller_rstd(src, True, eps, D, BLOCK)
    tl.store(Rstd + row * 2 + half, rstd)


@triton.jit
def _controller_down_grouped_kernel(
    Dest,
    Message,
    Down,
    Partials,
    Rstd,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP: tl.constexpr,
    PREPASS: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    first = tl.program_id(1).to(tl.int64) * GROUP
    s = tl.program_id(2).to(tl.int64)
    half = g // (D // BK)
    offset = (g % (D // BK)) * BK + tl.arange(0, BK)
    src = tl.where(half == 0, Dest + s * stride_dest_s, Message + s * T * D)
    lanes = tl.arange(0, 16)
    t = first + lanes
    mask = (lanes < GROUP) & (t < T)
    if PREPASS:
        rstd = tl.load(Rstd + (s * T + t) * 2 + half, mask, other=0)
        values = tl.load(src + offset[:, None] + t[None, :] * D, mask[None, :], other=0)
        x = (values * rstd[None, :]).to(tl.bfloat16)
    else:
        x = tl.full((BK, 16), 0, tl.bfloat16)
        for i in tl.static_range(GROUP):
            row = src + (first + i) * D
            rstd = _controller_rstd(row, first + i < T, eps, D, BLOCK)
            values = tl.load(row + offset, first + i < T, other=0)
            x = tl.where(
                lanes[None, :] == i, (values * rstd)[:, None].to(tl.bfloat16), x
            )
    ranks = tl.arange(0, R)
    ks = g * BK + tl.arange(0, BK)
    w = tl.load(Down + s * (R * 2 * D) + ranks[:, None] * (2 * D) + ks[None, :])
    partials = tl.dot(w, x)
    dst = Partials + ((s * T + t) * (2 * D // BK) + g) * R
    tl.store(dst[None, :] + ranks[:, None], partials, mask[None, :])


@triton.jit
def _controller_blend_grouped_kernel(
    Dest,
    Message,
    Partials,
    GainForget,
    Mixed,
    SumSq,
    stride_dest_s,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
    G_POW2: tl.constexpr,
    BN: tl.constexpr,
    GROUP: tl.constexpr,
):
    c = tl.program_id(0).to(tl.int64)
    first = tl.program_id(1).to(tl.int64) * GROUP
    s = tl.program_id(2).to(tl.int64)
    lanes = tl.arange(0, 16)
    t = first + lanes
    row = s * T + t
    mask = (lanes < GROUP) & (t < T)
    ranks = tl.arange(0, R)
    slices = tl.arange(0, G_POW2)
    hidden = tl.full((R, 16), 0, tl.bfloat16)
    for i in tl.static_range(GROUP):
        part = (s * T + first + i) * G + slices
        values = tl.load(
            Partials + part[:, None] * R + ranks[None, :],
            (slices[:, None] < G) & (first + i < T),
            other=0,
        )
        h = tl.sum(values, 0).to(tl.bfloat16)
        hidden = tl.where(lanes[None, :] == i, h[:, None], hidden)
    both = tl.arange(0, 2 * BN)
    out_rows = c * BN + both % BN + tl.where(both >= BN, D, 0)
    weights = GainForget + s * (2 * D * R) + out_rows[:, None] * R + ranks[None, :]
    product = tl.dot(tl.load(weights), hidden)
    gain, forget = tl.split(tl.permute(tl.reshape(product, (2, BN, 16)), (1, 2, 0)))
    cols = c * BN + tl.arange(0, BN)
    dest = tl.load(
        Dest + s * stride_dest_s + t[None, :] * D + cols[:, None],
        mask[None, :],
        other=0,
    )
    dst = row[None, :] * D + cols[:, None]
    message = tl.load(Message + dst, mask[None, :], other=0)
    x = (1.0 - tldevice.tanh(forget)) * dest + tldevice.tanh(gain) * message
    tl.store(Mixed + dst, x, mask[None, :])
    tl.store(SumSq + row * (D // BN) + c, tl.sum(x * x, axis=0), mask)


@triton.jit
def _controller_norm_kernel(
    Mixed,
    SumSq,
    NormW,
    Out,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    C: tl.constexpr,
    C_POW2: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    row = s * T + t
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    chunks = tl.arange(0, C_POW2)
    sum_sq = tl.sum(tl.load(SumSq + row * C + chunks, chunks < C, other=0.0))
    rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
    y = tl.load(Mixed + row * D + cols, mask, other=0) * rstd
    y = y * tl.load(NormW + s * stride_w_s + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * D + cols, y.to(Out.dtype.element_ty), mask)


def controller_blend_supported(hidden: int, rank: int) -> bool:
    """Whether `controller_blend` covers controllers of this width and rank."""
    return (
        hidden % _BLEND_SLICE == 0
        and (2 * hidden) % _DOWN_SLICE == 0
        and rank >= 16
        and rank & (rank - 1) == 0
    )


def controller_blend(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """`blend_rms_norm` of the controllers `down` bf16 `(S, R, 2D)` and
    `gain_forget` bf16 `(S, 2D, R)` on fp32 `(S, T, D)` `dest` (may broadcast
    over streams) and `message`, for a few tokens: after `pair_rms_norm`,
    one launch per stage (down projection by column slices, blend by column
    slices, RMSNorm by row) instead of cuBLAS GEMMs per stream."""
    num_streams, num_tokens, d = message.shape
    rank = down.shape[1]
    assert down.shape == (num_streams, rank, 2 * d) and down.is_contiguous()
    assert gain_forget.shape == (num_streams, 2 * d, rank)
    assert gain_forget.is_contiguous()
    assert down.dtype == gain_forget.dtype == torch.bfloat16
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    assert controller_blend_supported(d, rank)
    # The kernels index both inputs as `(s * T + t) * D` rows.
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    num_slices = 2 * d // _DOWN_SLICE
    partials = message.new_empty((num_streams, num_tokens, num_slices, rank))
    grid = (num_slices, num_tokens, num_streams)
    fused_norm = _fused_down_norm() and d % _DOWN_SLICE == 0
    group, prepass = _grouped_controller()
    grouped = group > 1 and 1 < num_tokens <= CONTROLLER_MAX_TOKENS and fused_norm
    if grouped:
        rstd = message
        if prepass:
            rstd = message.new_empty((num_streams, num_tokens, 2))
            _controller_rstd_kernel[(num_tokens, num_streams, 2)](
                dest,
                message,
                rstd,
                dest.stride(0),
                num_tokens,
                1e-6,
                D=d,
                BLOCK=triton.next_power_of_2(d),
                num_warps=_DOWN_WARPS,
            )
        tiles = triton.cdiv(num_tokens, group)
        _controller_down_grouped_kernel[(num_slices, tiles, num_streams)](
            dest,
            message,
            down,
            partials,
            rstd,
            dest.stride(0),
            num_tokens,
            1e-6,
            D=d,
            R=rank,
            BK=_DOWN_SLICE,
            BLOCK=triton.next_power_of_2(d),
            GROUP=group,
            PREPASS=prepass,
            num_warps=_DOWN_WARPS,
            num_stages=1,
        )
    elif fused_norm:
        _controller_down_norm_kernel[grid](
            dest,
            message,
            down,
            partials,
            dest.stride(0),
            num_tokens,
            1e-6,
            D=d,
            R=rank,
            BK=_DOWN_SLICE,
            BLOCK=triton.next_power_of_2(d),
            num_warps=_DOWN_WARPS,
            num_stages=1,
        )
    else:
        pair = pair_rms_norm(dest, message)
        _controller_down_kernel[grid](
            pair,
            down,
            partials,
            pair.stride(0),
            num_tokens,
            D2=2 * d,
            R=rank,
            BK=_DOWN_SLICE,
            num_warps=_DOWN_WARPS,
            num_stages=1,
        )
    num_chunks = d // _BLEND_SLICE
    mixed = torch.empty_like(message)
    sum_sq = message.new_empty((num_streams, num_tokens, num_chunks))
    if grouped:
        _controller_blend_grouped_kernel[(num_chunks, tiles, num_streams)](
            dest,
            message,
            partials,
            gain_forget,
            mixed,
            sum_sq,
            dest.stride(0),
            num_tokens,
            D=d,
            R=rank,
            G=num_slices,
            G_POW2=triton.next_power_of_2(num_slices),
            BN=_BLEND_SLICE,
            GROUP=group,
            num_warps=_BLEND_WARPS,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        _controller_blend_kernel[(num_chunks, num_tokens, num_streams)](
            dest,
            message,
            partials,
            gain_forget,
            mixed,
            sum_sq,
            dest.stride(0),
            num_tokens,
            D=d,
            R=rank,
            G=num_slices,
            G_POW2=triton.next_power_of_2(num_slices),
            BN=_BLEND_SLICE,
            num_warps=_BLEND_WARPS,
            num_stages=1,
            # The blend is unfused (no FMA), as the reference.
            enable_fp_fusion=False,
        )
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    _controller_norm_kernel[(num_streams, num_tokens)](
        mixed,
        sum_sq,
        norm_weight,
        out,
        norm_weight.stride(0),
        num_tokens,
        norm_eps,
        D=d,
        BLOCK=triton.next_power_of_2(d),
        C=num_chunks,
        C_POW2=triton.next_power_of_2(num_chunks),
        num_warps=8,
    )
    return out
