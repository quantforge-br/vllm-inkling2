# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8-INT8 prefill GEMM (`VLLM_INKLING2_PREFILL_INT8_ACT=1`).

Symmetric int4 weights (`uint4b8`, one positive scale per 128 inputs) times
activations quantized per token to symmetric int8, on the int8 tensor cores
(SM86: 4x the bf16 rate with fp32 accumulation). Only batches of at least
`VLLM_INKLING2_PREFILL_INT8_MIN_M` rows (prefill chunks) take it; decode and
speculative verification keep the exact W4A16 Marlin path.

Weight layout (`prepare_w4a8`): uint8 `[N, K / 2]` (K contiguous, as the
Ampere int8 MMA operands must be), where byte `64 g + r` of row n holds code
`128 g + r` in its low nibble and code `128 g + 64 + r` in its high nibble, so
each 128-wide group is two int8 MMAs over 64 inputs whose int32 sum takes the
group's scale.
"""

import functools
import os

import torch

from vllm.triton_utils import tl, triton

GROUP = 128


@functools.cache
def prefill_int8_act_enabled() -> bool:
    return os.environ.get("VLLM_INKLING2_PREFILL_INT8_ACT", "0") == "1"


@functools.cache
def prefill_int8_min_m() -> int:
    return int(os.environ.get("VLLM_INKLING2_PREFILL_INT8_MIN_M", "64"))


def prepare_w4a8(
    codes: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(weight uint8 [N, K / 2], scales fp32 [K / GROUP, N])` from uint4b8
    codes `[N, K]` (q + 8) and their group scales `[N, K / GROUP]`."""
    n, k = codes.shape
    assert k % GROUP == 0 and scales.shape == (n, k // GROUP)
    # Zero scales are the zero rows that pad fused projections.
    assert bool((scales >= 0).all()), "W4A8 needs non-negative scales"
    blocks = codes.view(n, k // GROUP, 2, GROUP // 2).to(torch.uint8)
    packed = blocks[:, :, 0] | (blocks[:, :, 1] << 4)  # [N, K/128, 64]
    weight = packed.reshape(n, k // 2).contiguous()
    return weight, scales.float().t().contiguous()


@triton.jit
def _quantize_rows_kernel(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(X + row * K + cols, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.where(amax > 0, amax / 127.0, 1.0)
    q = tl.extra.cuda.libdevice.rint(x / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(Q + row * K + cols, q.to(tl.int8), mask=mask)
    tl.store(S + row, scale)


def quantize_rows_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8 `x` and its fp32 row scales."""
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    if m:
        _quantize_rows_kernel[(m,)](
            x, q, s, K=k, BLOCK=triton.next_power_of_2(k), num_warps=8
        )
    return q, s


def _configs():
    return [
        triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=s)
        for bm, bn, w, s in (
            (128, 128, 4, 3),
            (128, 128, 8, 3),
            (128, 256, 8, 3),
            (64, 128, 4, 4),
            (64, 256, 4, 3),
            (128, 64, 4, 4),
            (256, 128, 8, 3),
        )
    ]


@triton.autotune(configs=_configs(), key=["N", "K", "M_BUCKET"])
@triton.jit
def _w4a8_kernel(
    A,  # int8 [M, K]
    AS,  # fp32 [M]
    B,  # uint8 [N, K / 2]
    BS,  # fp32 [K / GROUP, N]
    C,  # bf16 [M, N]
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    M_BUCKET: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    half = tl.arange(0, 64)
    row_mask = rows < M
    col_mask = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, K // 128):
        a_ptr = A + rows[:, None].to(tl.int64) * K + g * 128 + half[None, :]
        a_lo = tl.load(a_ptr, mask=row_mask[:, None], other=0)
        a_hi = tl.load(a_ptr + 64, mask=row_mask[:, None], other=0)
        b = tl.load(
            B + cols[:, None].to(tl.int64) * (K // 2) + g * 64 + half[None, :],
            mask=col_mask[:, None],
            other=0x88,
        )
        lo = ((b & 0xF).to(tl.int16) - 8).to(tl.int8)
        hi = ((b >> 4).to(tl.int16) - 8).to(tl.int8)
        part = tl.dot(a_lo, tl.trans(lo), out_dtype=tl.int32)
        part = tl.dot(a_hi, tl.trans(hi), acc=part, out_dtype=tl.int32)
        scale = tl.load(BS + g * N + cols, mask=col_mask, other=0.0)
        acc += part.to(tl.float32) * scale[None, :]
    row_scale = tl.load(AS + rows, mask=row_mask, other=0.0)
    out = acc * row_scale[:, None]
    tl.store(
        C + rows[:, None].to(tl.int64) * N + cols[None, :],
        out.to(C.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def w4a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """bf16 `x [M, K] @ W.T` for a `prepare_w4a8` weight."""
    m, k = x.shape
    n = weight.shape[0]
    assert weight.shape[1] * 2 == k
    q, s = quantize_rows_int8(x.contiguous())
    if out is None:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    bucket = min(triton.next_power_of_2(max(m, 1)), 4096)

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))

    _w4a8_kernel[grid](q, s, weight, scales, out, m, N=n, K=k, M_BUCKET=bucket)
    return out
