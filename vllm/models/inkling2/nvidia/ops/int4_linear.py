# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only int4 linears (W4A16) for the decode GEMVs of GPUs without the
Hopper / Blackwell low-M kernels.

A packed weight is uint8 `[N, K / 2 + 4 * K / GROUP + pad]`: row `n` holds the
asymmetric 4-bit codes of output row `n` (two per byte, low nibble first),
then one fp16 `(step, low)` pair per `GROUP` input columns, so that
`w = code * step + low`. Rows are 16-byte aligned and row slices keep their
group parameters.
"""

import torch

from vllm.triton_utils import tl, triton

GROUP = 128
_MAX_M = 16


def is_packed_int4(weight: torch.Tensor) -> bool:
    # No other Inkling2 weight is stored as uint8.
    return weight.dtype == torch.uint8


def packed_int4_row_bytes(k: int) -> int:
    return -(-(k // 2 + 4 * (k // GROUP)) // 16) * 16


def pack_int4_rows(weight: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest asymmetric 4-bit codes of `weight` `[N, K]` per
    `GROUP` columns of each output row, packed with their fp16 parameters."""
    n, k = weight.shape
    assert k % (2 * GROUP) == 0
    groups = k // GROUP
    w = weight.float().view(n, groups, GROUP)
    low = w.amin(-1, keepdim=True).to(torch.float16).float()
    step = ((w.amax(-1, keepdim=True) - low) / 15).to(torch.float16).float()
    step = torch.where(step > 0, step, torch.ones_like(step))
    codes = torch.round((w - low) / step).clamp_(0, 15).to(torch.uint8).view(n, k)
    packed = torch.zeros(
        (n, packed_int4_row_bytes(k)), dtype=torch.uint8, device=weight.device
    )
    packed[:, : k // 2] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    params = torch.cat((step, low), -1).to(torch.float16)
    packed[:, k // 2 : k // 2 + 4 * groups] = params.view(torch.uint8).view(n, -1)
    return packed


def unpack_int4_rows(packed: torch.Tensor, k: int) -> torch.Tensor:
    """The fp32 `[N, K]` weight a `pack_int4_rows` tensor stands for."""
    n, groups = packed.shape[0], k // GROUP
    codes = torch.stack((packed[:, : k // 2] & 15, packed[:, : k // 2] >> 4), -1)
    params = packed[:, k // 2 : k // 2 + 4 * groups].contiguous()
    params = params.view(torch.float16).view(n, groups, 2).float()
    codes = codes.view(n, groups, GROUP).float()
    return (codes * params[..., :1] + params[..., 1:]).view(n, k)


@triton.jit
def _int4_gemv_kernel(
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
    G: tl.constexpr,
):
    # `BK` input columns (`BK / G` groups) per step: the codes are widened to
    # bf16 weights (`code * step + low`) and contracted by one `tl.dot`.
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    col_ok = cols < N
    w_rows = W + cols.to(tl.int64) * SW
    params = (w_rows + K // 2).to(tl.pointer_type(tl.float16))
    halves = tl.arange(0, BK // 2)
    groups = tl.arange(0, BK // G)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        packed = tl.load(
            w_rows[None, :] + (k0 // 2 + halves)[:, None], col_ok[None, :], other=0
        )
        codes = tl.join(packed & 15, packed >> 4)  # [BK / 2, BN, 2]
        codes = tl.reshape(tl.permute(codes, (0, 2, 1)), (BK // G, G, BN))
        pair = params[None, :] + 2 * (k0 // G + groups)[:, None]
        step = tl.load(pair, col_ok[None, :], other=0).to(tl.float32)
        low = tl.load(pair + 1, col_ok[None, :], other=0).to(tl.float32)
        w = codes.to(tl.float32) * step[:, None, :] + low[:, None, :]
        ks = k0 + tl.arange(0, BK)
        x = tl.load(X + rows[:, None] * SX + ks[None, :], rows[:, None] < M, other=0)
        acc += tl.dot(x, tl.reshape(w, (BK, BN)).to(x.dtype))
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & col_ok[None, :],
    )


def int4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`x @ dequant(weight).T` with fp32 accumulation for bf16 `x` `[..., K]`
    and a `pack_int4_rows` weight; more than 16 rows run 16 at a time."""
    k = x.shape[-1]
    assert weight.shape[1] == packed_int4_row_bytes(k) and k % 256 == 0
    assert x.stride(-1) == 1 and weight.stride(-1) == 1 and weight.stride(0) % 16 == 0
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m, n = x2.shape[0], weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    else:
        assert out.shape == (m, n) and out.stride(-1) == 1
    for start in range(0, m, _MAX_M):
        rows = min(_MAX_M, m - start)
        _int4_gemv_kernel[(triton.cdiv(n, 32),)](
            x2[start:],
            weight,
            out[start:],
            rows,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            BN=32,
            BK=256,
            MP=16,
            G=GROUP,
            num_warps=4,
            # Tiles from a sweep on RTX 3090 (N 201024 / 8600, K 2304).
            num_stages=3,
        )
    return out.view(*lead, n)
