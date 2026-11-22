# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPTQ for the Marlin W4A16 attention projections.

Two phases, both inside the serving engine so the weights keep this model's
tensor-parallel layout:

1. `VLLM_INKLING2_GPTQ_COLLECT=<dir>`: every eager (prefill) projection adds
   `x^T x` of its input to a per-layer, per-input fp32 Hessian; every
   `_SAVE_EVERY` tokens each rank writes them to `<dir>`.
2. `VLLM_INKLING2_GPTQ_DIR=<dir>` with `VLLM_INKLING2_DENSE_BITS=4`: at load
   time each projection with a saved Hessian is quantized by GPTQ (asymmetric
   uint4, groups of 128, no activation reordering) instead of
   round-to-nearest."""

import functools
import os

import torch

from .marlin_linear import GROUP

_SAVE_EVERY = 16384
_DAMP = 0.01
_BLOCK = 128


@functools.cache
def collect_dir() -> str | None:
    return os.environ.get("VLLM_INKLING2_GPTQ_COLLECT") or None


@functools.cache
def hessian_dir() -> str | None:
    return os.environ.get("VLLM_INKLING2_GPTQ_DIR") or None


def _path(root: str, name: str, rank: int) -> str:
    return os.path.join(root, f"rank{rank}", name.replace("/", "_") + ".pt")


class _Collector:
    def __init__(self) -> None:
        self.hessians: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.tokens = 0
        self.saved_at = 0
        self.first: str | None = None
        self.active = False

    def add(self, name: str, x: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing() or x.shape[0] == 0:
            return
        if self.first is None:
            self.first = name
        if name == self.first:
            # Only while `<dir>/ACTIVE` exists: keeps warmup batches out.
            root = collect_dir()
            assert root is not None
            was_active = self.active
            self.active = os.path.exists(os.path.join(root, "ACTIVE"))
            if self.active:
                self.step(x.reshape(-1, x.shape[-1]).shape[0])
            elif was_active:
                self.save()  # final flush on the first call after `ACTIVE` goes
        if not self.active:
            return
        x = x.reshape(-1, x.shape[-1]).float()
        h = self.hessians.get(name)
        if h is None:
            h = self.hessians[name] = torch.zeros(
                (x.shape[1], x.shape[1]), dtype=torch.float32, device=x.device
            )
            self.counts[name] = 0
        h.addmm_(x.t(), x)
        self.counts[name] += x.shape[0]

    def step(self, tokens: int) -> None:
        self.tokens += tokens
        if self.tokens - self.saved_at < _SAVE_EVERY:
            return
        self.saved_at = self.tokens
        self.save()

    def save(self) -> None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        rank = get_tensor_model_parallel_rank()
        root = collect_dir()
        assert root is not None
        os.makedirs(os.path.join(root, f"rank{rank}"), exist_ok=True)
        for name, h in self.hessians.items():
            path = _path(root, name, rank)
            saved = {
                "h": h.cpu(),
                "n": self.counts[name],
                "tp_size": get_tensor_model_parallel_world_size(),
            }
            torch.save(saved, path + ".tmp")
            os.replace(path + ".tmp", path)


COLLECTOR = _Collector()


def load_hessian(
    name: str, rank: int, tp_size: int, k: int
) -> torch.Tensor | None:
    """The mean `x^T x` `[k, k]` saved for `name` on `rank` (None: no file)."""
    root = hessian_dir()
    if root is None:
        return None
    path = _path(root, name, rank)
    if not os.path.exists(path):
        return None
    saved = torch.load(path, map_location="cpu")
    h, n = saved["h"], saved["n"]
    if saved.get("tp_size") != tp_size or h.shape != (k, k) or n <= 0:
        raise ValueError(
            f"GPTQ Hessian {path}: tp_size {saved.get('tp_size')}, shape "
            f"{tuple(h.shape)}, {n} tokens; expected tp_size {tp_size}, ({k}, {k})"
        )
    if not torch.isfinite(h).all():
        raise ValueError(f"GPTQ Hessian {path} is not finite")
    return h / n


def _group_params(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16-rounded scale and integer zero of `w` `[N, GROUP]` per row."""
    lo = w.amin(dim=1).clamp(max=0)
    hi = w.amax(dim=1).clamp(min=0)
    scale = ((hi - lo) / 15).clamp(min=1e-8).bfloat16().float()
    zero = torch.round(-lo / scale).clamp(0, 15)
    return scale, zero


@torch.no_grad()
def gptq_w4_asym(
    weight: torch.Tensor, hessian: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ codes `[N, K]`, bf16 scales and int zero points `[N, K/GROUP]` of
    `weight` `[N, K]` for the input Hessian `hessian` `[K, K]` (the layout of
    `marlin_linear.quantize_w4_asym`)."""
    n, k = weight.shape
    assert k % GROUP == 0 and _BLOCK % GROUP == 0
    w = weight.float().clone()
    h = hessian.to(w.device, torch.float32).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(_DAMP * torch.mean(torch.diag(h)))
    hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True
    )
    codes = torch.empty((n, k), dtype=torch.uint8, device=w.device)
    scales = torch.empty((n, k // GROUP), dtype=torch.float32, device=w.device)
    zeros = torch.empty((n, k // GROUP), dtype=torch.float32, device=w.device)
    for i1 in range(0, k, _BLOCK):
        i2 = min(i1 + _BLOCK, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % GROUP == 0:
                g = col // GROUP
                # The group's parameters come from its error-updated weights.
                group = torch.cat((w1[:, i:], w[:, i2 : col + GROUP]), dim=1)
                scales[:, g], zeros[:, g] = _group_params(group[:, :GROUP])
            g = col // GROUP
            s, z = scales[:, g], zeros[:, g]
            q = torch.clamp(torch.round(w1[:, i] / s) + z, 0, 15)
            codes[:, col] = q.to(torch.uint8)
            err = (w1[:, i] - (q - z) * s) / hinv1[i, i]
            w1[:, i:] -= err.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
            err1[:, i] = err
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return codes, scales.bfloat16(), zeros.to(torch.int32)


def quantize_projection(
    name: str, weight: torch.Tensor, row_blocks: list[tuple[str, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """GPTQ of `weight` `[N, K]` whose consecutive row blocks `(input, rows)`
    read the inputs `<name>.<input>` (None: some Hessian is missing)."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    assert sum(rows for _, rows in row_blocks) == weight.shape[0]
    parts, start = [], 0
    for key, rows in row_blocks:
        h = load_hessian(f"{name}.{key}", rank, tp_size, weight.shape[1])
        if h is None:
            return None
        parts.append(gptq_w4_asym(weight[start : start + rows], h))
        start += rows
    return tuple(torch.cat(p) for p in zip(*parts))  # type: ignore[return-value]
