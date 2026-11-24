# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 dense projections on vLLM's Marlin GEMM (asymmetric uint4, groups of
128 input columns, bf16 scales and per-group zero points).

On SM86 Marlin reads a 4-bit weight about twice as fast as the int8 Triton
GEMVs of `int8_linear` at 1..24 rows, so it replaces them for the attention
projections when `VLLM_INKLING2_DENSE_BITS=4`."""

import functools
import os

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_workspace,
    marlin_permute_scales,
    marlin_zero_points,
)
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton

GROUP = 128
_N_ALIGN = 64
_QTYPE = scalar_types.uint4


@functools.cache
def dense_bits() -> int:
    """Weight bits of the attention projections in the int8 dense mode
    (`VLLM_INKLING2_DENSE_BITS`: 8 W8A16 (default), 4 Marlin W4A16)."""
    bits = int(os.environ.get("VLLM_INKLING2_DENSE_BITS", "8"))
    assert bits in (4, 8), bits
    return bits


def is_packed_marlin(weight: torch.Tensor) -> bool:
    return getattr(weight, "marlin_n", None) is not None


def quantize_w4_asym(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-to-nearest asymmetric 4-bit codes of `weight` `[N, K]`: codes
    `[N, K]` in 0..15, bf16 scales and integer zero points `[N, K / GROUP]`,
    with `w ~= (code - zero) * scale`."""
    n, k = weight.shape
    assert k % GROUP == 0, k
    w = weight.float().view(n, k // GROUP, GROUP)
    lo = w.amin(dim=-1).clamp(max=0)
    hi = w.amax(dim=-1).clamp(min=0)
    scale = ((hi - lo) / 15).clamp(min=1e-8).bfloat16()
    zero = torch.round(-lo / scale.float()).clamp(0, 15)
    codes = torch.round(w / scale.float().unsqueeze(-1)) + zero.unsqueeze(-1)
    return codes.clamp_(0, 15).to(torch.uint8).view(n, k), scale, zero.to(torch.int32)


def dequantize_w4(
    codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    """The fp32 `[N, K]` weight of `quantize_w4_asym` codes."""
    n, k = codes.shape
    w = codes.float().view(n, k // GROUP, GROUP) - zero.float().unsqueeze(-1)
    return (w * scale.float().unsqueeze(-1)).view(n, k)


def pack_marlin_w4(
    codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    """Marlin tensor for codes `[N, K]`, scales and zero points `[N, K/GROUP]`.
    N is padded to a multiple of 64 with zero columns; the scales and zero
    points ride on the returned int32 weight as attributes."""
    n, k = codes.shape
    pad = -n % _N_ALIGN
    if pad:
        codes = torch.nn.functional.pad(codes, (0, 0, 0, pad))
        scale = torch.nn.functional.pad(scale, (0, 0, 0, pad))
        zero = torch.nn.functional.pad(zero, (0, 0, 0, pad))
    size_n = n + pad
    groups = k // GROUP
    # GPTQ layout (eight K-consecutive codes per int32), then Marlin's repack.
    nibbles = codes.t().contiguous().to(torch.int32).view(k // 8, 8, size_n)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=codes.device)
    gptq = (nibbles << shifts.view(1, 8, 1)).sum(dim=1, dtype=torch.int32)
    q = ops.gptq_marlin_repack(gptq, k, size_n, 4)
    q.marlin_n = n
    q.marlin_k = k
    q.marlin_scale = marlin_permute_scales(scale.t().contiguous(), k, size_n, GROUP).to(
        codes.device
    )
    q.marlin_zero = marlin_zero_points(
        zero.t().contiguous().cpu(), groups, size_n, 4
    ).to(codes.device)
    return q


def marlin_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`x @ dequant(weight).T` for bf16 `x` `[..., K]` and a `pack_marlin_w4`
    weight (fp32 accumulation; `out_dtype` other than `x.dtype` converts)."""
    n, k = weight.marlin_n, weight.marlin_k
    size_n = weight.marlin_scale.shape[1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    direct = out is not None and size_n == n and out.is_contiguous()
    if out is not None:
        assert out.shape == (m, n)
    if m == 0:
        res = out if out is not None else x.new_empty((m, n), dtype=out_dtype)
        return res.view(*lead, n)
    y = ops.marlin_gemm(
        x2,
        out if direct and out.dtype == x.dtype else None,
        weight,
        None,
        weight.marlin_scale,
        None,
        weight.marlin_zero,
        get_marlin_workspace(x.device),
        getattr(weight, "marlin_type", _QTYPE),
        size_m=m,
        size_n=size_n,
        size_k=k,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    if size_n != n:
        y = y[:, :n]
    if out is not None:
        if y.data_ptr() != out.data_ptr():
            out.copy_(y)
        return out.view(*lead, n)
    if out_dtype is not None and out_dtype != y.dtype:
        y = y.to(out_dtype)
    if len(lead) == 1:
        return y  # a padded weight leaves a strided view, as `split` would
    return y.reshape(*lead, n)


_ATTRS = ("marlin_n", "marlin_k", "marlin_scale", "marlin_zero")


def marlin_parameter(
    weight: torch.Tensor,
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> nn.Parameter:
    """Marlin W4A16 parameter of a bf16 `[N, K]` weight: round-to-nearest, or
    the given `(codes, scales, zeros)` (e.g. from GPTQ)."""
    packed = pack_marlin_w4(*(quantized or quantize_w4_asym(weight)))
    param = nn.Parameter(packed, requires_grad=False)
    for name in _ATTRS:
        setattr(param, name, getattr(packed, name))
    return param


def prepare_marlin_w8(source: torch.Tensor, packed: torch.Tensor) -> None:
    """Derive a Marlin shard from the same signed codes as the Triton GEMV.

    The public source retains its physical row layout for vocab indexing and
    refolding. Keep captured device pointers stable on a weight refresh.
    """
    from .int8_linear import ROW_PAD

    n, stride = packed.shape
    k = stride - ROW_PAD
    assert packed.dtype == torch.int8 and k % 16 == 0
    padded_n = triton.cdiv(n, _N_ALIGN) * _N_ALIGN
    codes = torch.full((padded_n, k), 128, dtype=torch.uint8, device=packed.device)
    torch.bitwise_xor(packed[:, :k].view(torch.uint8), 128, out=codes[:n])
    gptq = codes.view(torch.int32).t().contiguous()
    weight = ops.gptq_marlin_repack(gptq, k, padded_n, 8)
    row_scales = packed[:, k : k + 4].contiguous().view(torch.float32).bfloat16()
    scales = torch.nn.functional.pad(row_scales.t(), (0, padded_n - n))
    scales = marlin_permute_scales(scales, k, padded_n, -1)
    existing = getattr(source, "marlin_w8", None)
    if existing is not None:
        existing.copy_(weight)
        existing.marlin_scale.copy_(scales)
        source.marlin_row_scales.copy_(row_scales)
        return
    weight.marlin_n = n
    weight.marlin_k = k
    weight.marlin_type = scalar_types.uint8b128
    weight.marlin_scale = scales
    weight.marlin_zero = None
    source.marlin_row_scales = row_scales
    source.marlin_w8 = weight
