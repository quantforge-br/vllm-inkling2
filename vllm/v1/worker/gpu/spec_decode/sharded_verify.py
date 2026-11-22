# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative verification at temperature 1.0 from tensor-parallel
vocabulary shards of the target logits, without all-gathering them.

For one-hot drafts with unprocessed logits, `rejection_sample` needs per
logits row only the log-sum-exp, the logit of the next draft token and the
Gumbel-max winner of the residual (the target distribution without that
draft token). Each rank reduces those over its shard, one small all-reduce
gathers them, and every rank runs the same acceptance loop. The Gumbel noise
is keyed by the global token id, seed and position exactly as in
`_resample_kernel`, so the samples follow the same random streams.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _is_bonus_logit

# Max, summed exponentials, draft-token logit, Gumbel max, Gumbel argmax.
NUM_STATS = tl.constexpr(5)
# Marks NaN winners in the token-id slot: above every token id, and ids up
# to 2 * NAN_SHIFT stay exact in fp32.
NAN_SHIFT = tl.constexpr(8388608.0)
_BLOCK = 4096


@triton.jit(do_not_specialize=["num_logits"])
def _shard_stats_kernel(
    X,
    stride_x,
    DraftSampled,
    Pos,
    ExpandedIdxMapping,
    ExpandedLocalPos,
    Seeds,
    Stats,
    start,
    n_valid,
    num_logits,
    splits,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    sp = tl.program_id(1)
    cols = sp * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_valid
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    m = tl.max(x, axis=0)
    s = tl.where(m > float("-inf"), tl.sum(tl.exp(x - m)), 0.0)
    # The draft token a row verifies is the next row's input; a request's
    # bonus row verifies none.
    draft = tl.load(
        DraftSampled + row + 1,
        mask=not _is_bonus_logit(ExpandedLocalPos, row, num_logits),
        other=-1,
    ).to(tl.int64)
    local_draft = tl.where(draft >= 0, draft - start, -1)
    is_draft = cols == local_draft
    xd = tl.max(tl.where(is_draft & mask, x, float("-inf")), axis=0)
    req_state_idx = tl.load(ExpandedIdxMapping + row).to(tl.int64)
    seed = tl.load(Seeds + req_state_idx)
    pos = tl.load(Pos + row)
    residual = tl.where(is_draft, float("-inf"), x)
    gv, gi = gumbel_noised_argmax(
        residual,
        start + cols,
        mask,
        seed,
        pos,
        1.0,
        IS_DRAFTING=False,
        USE_FP64=False,
        APPLY_TEMPERATURE=False,
    )
    # A block of only NaN logits has a NaN winner, which would fail every
    # equality below and leave no valid token. Rank it under every real
    # value, as rejection_sample does (any real logit, +inf included, beats
    # an all-NaN block), but above the -inf of empty splits. Its token id is
    # shifted up by NAN_SHIFT so a real score tied at -FLT_MAX wins the
    # lowest-id tie-break; the verify kernel unshifts the winner.
    gi = (start + sp * BLOCK + gi).to(tl.float32)
    is_nan = gv != gv
    gv = tl.where(is_nan, -3.4028234663852886e38, gv)
    gi = tl.where(is_nan, gi + NAN_SHIFT, gi)
    out = Stats + (row * splits + sp) * NUM_STATS
    tl.store(out + 0, m)
    tl.store(out + 1, s)
    tl.store(out + 2, xd)
    tl.store(out + 3, gv)
    tl.store(out + 4, gi)


@triton.jit
def _shard_finish_kernel(
    Stats, Packed, rank, rows, splits, TP: tl.constexpr, SPLITS: tl.constexpr
):
    # One row: combine the splits and write this rank's slot of the
    # `(NUM_STATS, TP, rows)` all-reduce buffer.
    row = tl.program_id(0).to(tl.int64)
    sp = tl.arange(0, SPLITS)
    valid = sp < splits
    base = Stats + (row * splits + sp) * NUM_STATS
    m = tl.load(base + 0, mask=valid, other=float("-inf"))
    s = tl.load(base + 1, mask=valid, other=0.0)
    xd = tl.load(base + 2, mask=valid, other=float("-inf"))
    gv = tl.load(base + 3, mask=valid, other=float("-inf"))
    gi = tl.load(base + 4, mask=valid, other=3.0e38)
    big_m = tl.max(m, axis=0)
    big_s = tl.where(
        big_m > float("-inf"),
        tl.sum(tl.where(m == float("-inf"), 0.0, s * tl.exp(m - big_m)), axis=0),
        0.0,
    )
    big_gv = tl.max(gv, axis=0)
    big_gi = tl.min(tl.where(gv == big_gv, gi, 3.0e38), axis=0)
    slot = Packed + rank * rows + row
    stride = TP * rows
    tl.store(slot + 0 * stride, big_m)
    tl.store(slot + 1 * stride, big_s)
    tl.store(slot + 2 * stride, tl.max(xd, axis=0))
    tl.store(slot + 3 * stride, big_gv)
    tl.store(slot + 4 * stride, big_gi)


@triton.jit
def _sharded_verify_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    Packed,
    draft_sampled_ptr,
    pos_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    seed_ptr,
    rows,
    TP: tl.constexpr,
):
    # `_rejection_kernel` + `_resample_kernel` + `_insert_resampled_kernel`
    # for one-hot drafts at temperature 1.0, from the gathered shard stats.
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    seed = tl.load(seed_ptr + req_state_idx)
    ranks = tl.arange(0, TP)
    stride = TP * rows
    accepted_length = tl.zeros((), tl.int64)
    verifying = tl.full((), 1, tl.int32)
    for i in range(end_idx - start_idx - 1):
        if verifying != 0:
            logit_idx = start_idx + i
            draft = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            # A -1 placeholder is a rejection with the full target residual.
            verifying = (draft >= 0).to(tl.int32)
            if verifying != 0:
                slot = Packed + ranks * rows + logit_idx
                m = tl.load(slot + 0 * stride)
                s = tl.load(slot + 1 * stride)
                big_m = tl.max(m, axis=0)
                lse = big_m + tl.log(
                    tl.sum(tl.where(m == float("-inf"), 0.0, s * tl.exp(m - big_m)))
                )
                target_logprob = tl.max(tl.load(slot + 2 * stride), axis=0) - lse
                u = tl_rand32(seed, tl.load(pos_ptr + logit_idx), includes_zero=False)
                accepted = target_logprob > tl.log(u)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft)
                accepted_length += accepted.to(tl.int64)
                verifying = accepted.to(tl.int32)
    # Resample the rejected (or bonus) row: Gumbel winner of its residual,
    # the lower rank (lower token ids) on a tie.
    slot = Packed + ranks * rows + start_idx + accepted_length
    gv = tl.load(slot + 3 * stride)
    gi = tl.load(slot + 4 * stride)
    best = tl.max(gv, axis=0)
    token = tl.min(tl.where(gv == best, gi, 3.0e38), axis=0)
    token = tl.where(token >= NAN_SHIFT, token - NAN_SHIFT, token)
    tl.store(
        sampled_ptr + req_idx * sampled_stride + accepted_length, token.to(tl.int64)
    )
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))


def shard_stats(
    logits: torch.Tensor,
    start: int,
    n_valid: int,
    rank: int,
    tp: int,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    seeds: torch.Tensor,
) -> torch.Tensor:
    """This rank's slot of the `(NUM_STATS, tp, rows)` all-reduce buffer for
    fp32 target logits `(rows, shard)` of the shard starting at `start`
    (columns at or past `n_valid` ignored)."""
    rows, width = logits.shape
    assert logits.stride(1) == 1
    assert start + logits.shape[1] <= 1 << 23, "token ids must stay below NAN_SHIFT"
    n_valid = max(0, min(width, n_valid))
    splits = max(1, triton.cdiv(n_valid, _BLOCK))
    stats = torch.empty(
        (rows, splits, NUM_STATS), dtype=torch.float32, device=logits.device
    )
    packed = torch.zeros((NUM_STATS, tp, rows), dtype=torch.float32, device=logits.device)
    if rows:
        _shard_stats_kernel[(rows, splits)](
            logits,
            logits.stride(0),
            draft_sampled,
            pos,
            expanded_idx_mapping,
            expanded_local_pos,
            seeds,
            stats,
            start,
            n_valid,
            rows,
            splits,
            BLOCK=_BLOCK,
            num_warps=8,
        )
        _shard_finish_kernel[(rows,)](
            stats,
            packed,
            rank,
            rows,
            splits,
            TP=tp,
            SPLITS=triton.next_power_of_2(splits),
            num_warps=1,
        )
    return packed


def sharded_verify(
    packed: torch.Tensor,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    seeds: torch.Tensor,
    num_reqs: int,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled tokens `(num_reqs, steps + 1)` and their int32 counts from the
    all-reduced shard stats."""
    _, tp, rows = packed.shape
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    _sharded_verify_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        packed,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        seeds,
        rows,
        TP=tp,
        num_warps=1,
    )
    return sampled, num_sampled
