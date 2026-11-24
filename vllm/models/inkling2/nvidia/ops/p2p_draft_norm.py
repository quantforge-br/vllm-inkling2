# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM86 TP2 draft all-reduce, residual update and RMSNorm, one CTA per row."""

import os

import torch

from vllm.distributed import get_tp_group
from vllm.triton_utils import tl, triton

from .p2p_conv import MAX_TOKENS, P2PConvWorkspace, _ranks_agree


@triton.jit
def _p2p_draft_norm_kernel(
    X,
    Residual,
    Weight,
    Output,
    Counter,
    recv_addr,
    peer_addr,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_T: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    gen = tl.load(Counter, volatile=True)
    base = (gen % 3) * MAX_T * D
    row = t * D + cols
    recv = recv_addr.to(tl.pointer_type(tl.bfloat16))
    peer = peer_addr.to(tl.pointer_type(tl.bfloat16))
    mine = tl.load(X + row, mask, other=0).to(tl.float32)
    mine = tl.where(mine == 0.0, 0.0, mine)
    tl.store(peer + base + row, mine.to(tl.bfloat16), mask)
    pending = 1
    while pending:
        landed = tl.load(recv + base + row, mask, other=0, volatile=True)
        pending = tl.sum((landed.to(tl.int16, bitcast=True) == -32768).to(tl.int32))
    other = tl.load(recv + base + row, mask, other=0, volatile=True).to(tl.float32)
    reduced = (mine + other).to(tl.bfloat16).to(tl.float32)
    total = tl.load(Residual + row, mask, other=0).to(tl.float32) + reduced
    tl.store(Residual + row, total.to(tl.bfloat16), mask)
    scale = tl.rsqrt(tl.sum(total * total) / D + eps)
    gamma = tl.load(Weight + cols, mask, other=0).to(tl.float32)
    tl.store(Output + row, total * scale * gamma, mask)

    # Re-arm all rows of the previous generation, including rows absent in
    # this (possibly narrower) batch, before advancing the device counter.
    stale = recv_addr.to(tl.pointer_type(tl.int16)) + ((gen + 2) % 3) * MAX_T * D
    num_rows = tl.num_programs(0)
    for i in tl.static_range(MAX_T):
        r = t + i * num_rows
        tl.store(
            stale + r * D + cols,
            tl.full((BLOCK,), -32768, tl.int16),
            (r < MAX_T) & mask,
        )
    if tl.atomic_add(Counter + 1, 1) == num_rows - 1:
        tl.store(Counter + 1, 0)
        tl.store(Counter, gen + 1)


def draft_norm_workspace(hidden: int):
    group = get_tp_group()
    if group.world_size != 2:
        return None
    from . import use_portable_kernels

    enabled = (
        os.getenv("VLLM_INKLING2_P2P_AR_DRAFT_NORM") == "1"
        and torch.cuda.get_device_capability() == (8, 6)
        and use_portable_kernels()
        and torch.cuda.can_device_access_peer(group.local_rank, 1 - group.local_rank)
    )
    # Even mismatched environment flags must resolve to the same decision.
    flags = [None, None]
    torch.distributed.all_gather_object(flags, enabled, group=group.cpu_group)
    if not all(flags) or not _ranks_agree():
        return None
    workspace = P2PConvWorkspace(hidden)
    dummy = torch.empty((MAX_TOKENS, hidden), device="cuda", dtype=torch.bfloat16)
    config = dict(
        D=hidden, BLOCK=triton.next_power_of_2(hidden), MAX_T=MAX_TOKENS, num_warps=8
    )
    args = (
        dummy,
        dummy,
        dummy[0],
        dummy,
        workspace.counter,
        workspace.recv_addr,
        workspace.peer_addr,
        1e-6,
    )
    try:
        workspace.check_residency(
            _p2p_draft_norm_kernel,
            (MAX_TOKENS, 1),
            args,
            config,
            ("draft_norm", hidden),
        )
        resident = True
    except RuntimeError:
        resident = False
    torch.distributed.all_gather_object(flags, resident, group=group.cpu_group)
    return workspace if all(flags) else None


def p2p_draft_norm(x, residual, weight, eps, workspace):
    tokens, hidden = x.shape
    assert workspace.usable(tokens, hidden)
    assert (
        x.shape == residual.shape
        and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    )
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    output = torch.empty_like(x)
    _p2p_draft_norm_kernel[(tokens, 1)](
        x,
        residual,
        weight,
        output,
        workspace.counter,
        workspace.recv_addr,
        workspace.peer_addr,
        eps,
        D=hidden,
        BLOCK=triton.next_power_of_2(hidden),
        MAX_T=MAX_TOKENS,
        num_warps=8,
    )
    return output
