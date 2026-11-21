# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only int8 linears (W8A16) for GPUs without the Hopper / Blackwell
low-M kernels.

A packed weight is int8 `[N, K + ROW_PAD]`: row `n` holds the symmetric codes
of output row `n` followed by its fp32 scale in the first 4 of `ROW_PAD`
bytes. Rows stay 16-byte aligned, and row slices of a fused projection keep
their scales.
"""

import functools
import os

import torch

from vllm.triton_utils import tl, triton

# Bytes after each row's codes; its fp32 scale takes the first 4.
ROW_PAD = 16
_GEMV_MAX_M = 16
# Output rows from which the row-major GEMV tile wins (the unembedding).
_WIDE_MIN_N = 65536


@functools.cache
def int8_dense_enabled() -> bool:
    """Whether to store the dense projections (attention, unembedding) as
    W8A16 int8 (`VLLM_INKLING2_INT8_DENSE=1`)."""
    return os.environ.get("VLLM_INKLING2_INT8_DENSE", "0") == "1"


@functools.cache
def head_bits() -> int:
    """Weight bits of the unembedding in the int8 dense mode
    (`VLLM_INKLING2_HEAD_BITS`: 16 keeps bf16, 8 W8A16 (default), 4 W4A16,
    see `int4_linear`)."""
    bits = int(os.environ.get("VLLM_INKLING2_HEAD_BITS", "8"))
    assert bits in (4, 8, 16), bits
    return bits


def is_packed_int8(weight: torch.Tensor) -> bool:
    return weight.dtype == torch.int8


def pack_int8_rows(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-row symmetric int8 codes of `weight` `[N, K]`, packed with
    their fp32 scales as `[N, K + ROW_PAD]`."""
    n, k = weight.shape
    assert k % 16 == 0
    w = weight.float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    codes = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
    packed = torch.zeros((n, k + ROW_PAD), dtype=torch.int8, device=weight.device)
    packed[:, :k] = codes
    packed[:, k : k + 4] = scale.view(torch.int8)
    return packed


def unpack_int8_rows(packed: torch.Tensor) -> torch.Tensor:
    """The fp32 `[N, K]` weight a `pack_int8_rows` tensor stands for."""
    k = packed.shape[1] - ROW_PAD
    scale = packed[:, k : k + 4].contiguous().view(torch.float32)
    return packed[:, :k].float() * scale


@triton.jit
def _int8_gemv_kernel(
    X,
    W,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    col_ok = cols < N
    w_rows = W + cols.to(tl.int64) * SW
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            w_rows[None, :] + ks[:, None],
            col_ok[None, :] & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w.to(x.dtype))
    scale = tl.load((w_rows + K).to(tl.pointer_type(tl.float32)), col_ok, other=0.0)
    acc = acc * scale[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & col_ok[None, :],
    )


@triton.jit
def _int8_gemv_wide_kernel(
    X,
    W,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    """`_int8_gemv_kernel` with the weight tile loaded along its rows: 15%
    faster on the 201k-row unembedding, slower on the small projections."""
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    col_ok = cols < N
    w_rows = W + cols.to(tl.int64) * SW
    acc = tl.zeros((BN, MP), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            w_rows[:, None] + ks[None, :],
            col_ok[:, None] & (ks[None, :] < K),
            other=0,
        )
        x = tl.load(
            X + rows[None, :] * SX + ks[:, None],
            (rows[None, :] < M) & (ks[:, None] < K),
            other=0,
        )
        acc += tl.dot(w.to(x.dtype), x)
    scale = tl.load((w_rows + K).to(tl.pointer_type(tl.float32)), col_ok, other=0.0)
    acc = acc * scale[:, None]
    tl.store(
        Y + rows[None, :] * SY + cols[:, None],
        acc.to(Y.dtype.element_ty),
        (rows[None, :] < M) & col_ok[:, None],
    )


@triton.jit
def _int8_gemv_slices_kernel(
    X,
    W,
    Y,
    M,
    K,
    SX,
    SW,
    SY,
    x1_offset,
    x2_offset,
    tiles0,
    tiles1,
    end0,
    end1,
    end2,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    """`_int8_gemv_kernel` over three consecutive row ranges of `W`
    (`[0, end0)`, `[end0, end1)`, `[end1, end2)`), each on its own input:
    `X`, and the rows `x1_offset` / `x2_offset` elements after it. Tiles do
    not cross a range."""
    pid = tl.program_id(0)
    in1 = (pid >= tiles0) & (pid < tiles0 + tiles1)
    in2 = pid >= tiles0 + tiles1
    start = tl.where(in2, end1, tl.where(in1, end0, 0))
    end = tl.where(in2, end2, tl.where(in1, end1, end0))
    tile = pid - tl.where(in2, tiles0 + tiles1, tl.where(in1, tiles0, 0))
    x_ptr = X + tl.where(in2, x2_offset, tl.where(in1, x1_offset, 0))
    cols = start + tile * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    col_ok = cols < end
    w_rows = W + cols.to(tl.int64) * SW
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            w_rows[None, :] + ks[:, None],
            col_ok[None, :] & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            x_ptr + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w.to(x.dtype))
    scale = tl.load((w_rows + K).to(tl.pointer_type(tl.float32)), col_ok, other=0.0)
    acc = acc * scale[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & col_ok[None, :],
    )


def int8_gemv_slices(
    xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    ends: tuple[int, int, int],
) -> torch.Tensor | None:
    """`[x0 @ W[:e0].T | x1 @ W[e0:e1].T | x2 @ W[e1:e2].T]` as one `[M, e2]`
    tensor from a single launch, for `M <= 16` rows of bf16 `[M, K]` inputs
    and a `pack_int8_rows` weight (None: not eligible)."""
    x0, x1, x2 = xs
    k = weight.shape[1] - ROW_PAD
    m = x0.shape[0]
    if not 0 < m <= _GEMV_MAX_M or x0.dim() != 2:
        return None
    for x in xs:
        if x.shape != (m, k) or x.dtype != x0.dtype or x.stride() != x0.stride():
            return None
        if x.device != x0.device:
            return None
    size = x0.element_size()
    offsets = [x.data_ptr() - x0.data_ptr() for x in (x1, x2)]
    if x0.stride(1) != 1 or any(offset % size for offset in offsets):
        return None
    assert weight.stride(-1) == 1 and weight.stride(0) % 16 == 0
    e0, e1, e2 = ends
    assert 0 < e0 < e1 < e2 <= weight.shape[0]
    out = torch.empty((m, e2), device=x0.device, dtype=x0.dtype)
    bn = 32
    tiles0, tiles1 = triton.cdiv(e0, bn), triton.cdiv(e1 - e0, bn)
    bk = 512 if k % 512 == 0 else 256
    _int8_gemv_slices_kernel[(tiles0 + tiles1 + triton.cdiv(e2 - e1, bn),)](
        x0,
        weight,
        out,
        m,
        k,
        x0.stride(0),
        weight.stride(0),
        out.stride(0),
        offsets[0] // size,
        offsets[1] // size,
        tiles0,
        tiles1,
        e0,
        e1,
        e2,
        BN=bn,
        BK=bk,
        MP=16,
        num_warps=4,
        num_stages=3 if bk == 512 else 2,
    )
    return out


@triton.jit
def _int8_gemm_kernel(
    X,
    W,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped launch order: consecutive programs share weight columns in L2.
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % (GROUP_M * num_n)) % group_m
    pid_n = (pid % (GROUP_M * num_n)) // group_m
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    row_ok = rows < M
    col_ok = cols < N
    x_rows = X + rows.to(tl.int64) * SX
    w_rows = W + cols.to(tl.int64) * SW
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(x_rows[:, None] + ks[None, :], row_ok[:, None], other=0)
        w = tl.load(w_rows[None, :] + ks[:, None], col_ok[None, :], other=0)
        acc += tl.dot(x, w.to(x.dtype))
    scale = tl.load((w_rows + K).to(tl.pointer_type(tl.float32)), col_ok, other=0.0)
    acc = acc * scale[None, :]
    tl.store(
        Y + rows[:, None].to(tl.int64) * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        row_ok[:, None] & col_ok[None, :],
    )


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`x @ dequant(weight).T` with fp32 accumulation for bf16/fp16 `x`
    `[..., K]` and a `pack_int8_rows` weight `[N, K + ROW_PAD]`."""
    k = weight.shape[1] - ROW_PAD
    assert x.shape[-1] == k and x.stride(-1) == 1 and weight.stride(-1) == 1
    assert weight.stride(0) % 16 == 0
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    else:
        assert out.shape == (m, n) and out.stride(-1) == 1
    if m == 0:
        return out.view(*lead, n)
    if m <= _GEMV_MAX_M and n >= _WIDE_MIN_N:
        _int8_gemv_wide_kernel[(triton.cdiv(n, 32),)](
            x2,
            weight,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            BN=32,
            BK=256,
            MP=16,
            num_warps=2,
            num_stages=1,
        )
    elif m <= _GEMV_MAX_M:
        # Tiles from a sweep on RTX 3090 (N 8576 / 2304, K 2304 / 3072).
        bk = 512 if k % 512 == 0 else 256
        _int8_gemv_kernel[(triton.cdiv(n, 32),)](
            x2,
            weight,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            BN=32,
            BK=bk,
            MP=16,
            num_warps=4,
            num_stages=3 if bk == 512 else 2,
        )
    else:
        assert k % 32 == 0
        wide = n > 4096
        bm = 32 if m <= 32 else 64 if m <= 64 or wide else 128
        bn = 128 if wide else 64
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
        _int8_gemm_kernel[grid](
            x2,
            weight,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            BM=bm,
            BN=bn,
            BK=32,
            GROUP_M=8,
            num_warps=4,
            num_stages=4,
        )
    return out.view(*lead, n)
