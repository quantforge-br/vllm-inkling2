# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPTQ for the Marlin W4A16 attention projections.

Two phases, both inside the serving engine so the weights keep this model's
tensor-parallel layout:

1. `VLLM_INKLING2_GPTQ_COLLECT=<dir>`: every eager (prefill) projection adds
   `x^T x` of its input to a per-layer, per-input fp32 Hessian; every
   `VLLM_INKLING2_GPTQ_SAVE_EVERY` tokens (0: only when collection stops)
   each rank writes them to `<dir>`. `VLLM_INKLING2_GPTQ_COLLECT_KINDS`
   (comma-separated `attn`, `experts`, `head`, `draft`; default `attn`)
   picks the projections, `VLLM_INKLING2_GPTQ_COLLECT_LAYERS` (e.g. `0-5,9`)
   the decoder layers whose experts are collected (one `[E, K, K]` Hessian
   per projection, each routed row weighted by its routing weight), and
   `VLLM_INKLING2_GPTQ_COLLECT_FULL=1` gathers the inputs of row-parallel
   projections so their Hessians span the unsharded input (`*.full`).
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


_KINDS = frozenset({"attn", "experts", "head", "draft"})


@functools.cache
def collect_kinds() -> frozenset[str]:
    if collect_dir() is None:
        return frozenset()
    spec = os.environ.get("VLLM_INKLING2_GPTQ_COLLECT_KINDS", "attn")
    kinds = frozenset(k for k in spec.split(",") if k)
    if kinds - _KINDS:
        raise ValueError(
            f"VLLM_INKLING2_GPTQ_COLLECT_KINDS: unknown {sorted(kinds - _KINDS)}"
        )
    return kinds


def collecting(kind: str) -> bool:
    return kind in collect_kinds()


@functools.cache
def collect_layers() -> frozenset[int] | None:
    spec = os.environ.get("VLLM_INKLING2_GPTQ_COLLECT_LAYERS")
    if not spec:
        return None
    layers: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        layers.update(range(int(lo), int(hi or lo) + 1))
    return frozenset(layers)


def collecting_layer_experts(layer: int) -> bool:
    layers = collect_layers()
    return collecting("experts") and (layers is None or layer in layers)


@functools.cache
def collect_full() -> bool:
    return os.environ.get("VLLM_INKLING2_GPTQ_COLLECT_FULL") == "1"


@functools.cache
def save_every() -> int:
    return int(os.environ.get("VLLM_INKLING2_GPTQ_SAVE_EVERY", str(_SAVE_EVERY)))


def _path(root: str, name: str, rank: int) -> str:
    return os.path.join(root, f"rank{rank}", name.replace("/", "_") + ".pt")


class _Collector:
    def __init__(self) -> None:
        self.hessians: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int | torch.Tensor] = {}
        self.tokens = 0
        self.saved_at = 0
        self.first: str | None = None
        self.active = False

    def _gate(self, name: str, tokens: int) -> bool:
        if torch.cuda.is_current_stream_capturing() or tokens == 0:
            return False
        if self.first is None:
            self.first = name
        if name == self.first:
            # Only while `<dir>/ACTIVE` exists: keeps warmup batches out.
            # Toggle it while the engine is idle, so every rank (and every
            # collective below) agrees.
            root = collect_dir()
            assert root is not None
            was_active = self.active
            self.active = os.path.exists(os.path.join(root, "ACTIVE"))
            if self.active:
                self.step(tokens)
            elif was_active:
                self.save()  # final flush on the first call after `ACTIVE` goes
        return self.active

    def add(self, name: str, x: torch.Tensor, gather: bool = False) -> None:
        """Adds `x^T x` of the rows of `x` to `name`'s Hessian; with
        `gather`, of the TP ranks' shards of `x`'s last dim, concatenated."""
        x = x.reshape(-1, x.shape[-1])
        if not self._gate(name, x.shape[0]):
            return
        if gather:
            from vllm.distributed import tensor_model_parallel_all_gather

            x = tensor_model_parallel_all_gather(x.contiguous(), dim=-1)
        x = x.float()
        h = self.hessians.get(name)
        if h is None:
            h = self.hessians[name] = torch.zeros(
                (x.shape[1], x.shape[1]), dtype=torch.float32, device=x.device
            )
            self.counts[name] = 0
        h.addmm_(x.t(), x)
        self.counts[name] += x.shape[0]

    def reserve_experts(
        self, name: str, num_experts: int, k: int, device: torch.device
    ) -> None:
        """Allocates `name`'s per-expert Hessians now (before the KV cache
        takes the free memory)."""
        if name not in self.hessians:
            self.hessians[name] = torch.zeros(
                (num_experts, k, k), dtype=torch.float32, device=device
            )
            self.counts[name] = torch.zeros(
                num_experts, dtype=torch.int64, device=device
            )

    def add_experts(
        self,
        name: str,
        x: torch.Tensor,
        ids: torch.Tensor,
        scale: torch.Tensor | None,
        num_experts: int,
        per_slot: bool,
    ) -> None:
        """Adds, per local expert, `x^T x` of the rows routed to it. `ids`
        `[T, topk]` are local expert ids (-1: another rank's); `x` holds one
        row per token (`[T, K]`) or per slot (`per_slot`, `[T * topk, K]`);
        `scale` `[T * topk]` weights each slot's row."""
        tokens, topk = ids.shape
        if not self._gate(name, tokens):
            return
        self.reserve_experts(name, num_experts, x.shape[-1], x.device)
        h = self.hessians[name]
        counts = self.counts[name]
        assert isinstance(counts, torch.Tensor)
        flat = ids.reshape(-1)
        slots = torch.nonzero(flat >= 0).squeeze(1)
        if slots.numel() == 0:
            return
        experts = flat[slots].long()
        rows = (x[slots] if per_slot else x[slots // topk]).float()
        if scale is not None:
            rows = rows * scale[slots].float().unsqueeze(1)
        order = torch.argsort(experts, stable=True)
        experts, rows = experts[order], rows[order]
        per_expert = torch.bincount(experts, minlength=num_experts)
        counts += per_expert
        start = 0
        for e, end in enumerate(per_expert.cumsum(0).tolist()):
            if end > start:
                block = rows[start:end]
                h[e].addmm_(block.t(), block)
            start = end

    def step(self, tokens: int) -> None:
        self.tokens += tokens
        every = save_every()
        if every <= 0 or self.tokens - self.saved_at < every:
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
            count = self.counts[name]
            saved = {
                "h": h.cpu(),
                "n": count.cpu() if isinstance(count, torch.Tensor) else count,
                "tp_size": get_tensor_model_parallel_world_size(),
            }
            if h.dim() == 3:
                # Global id of the first local expert (experts are split
                # evenly over the ranks).
                saved["expert_start"] = rank * h.shape[0]
            torch.save(saved, path + ".tmp")
            os.replace(path + ".tmp", path)
            del saved


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


# Scale candidates of the symmetric search: `amax / divisor * shrink`. The
# divisor 8 puts a negative extreme on code -8, 7 a positive one on +7.
_SYM_DIVISORS = (7.0, 7.5, 8.0)
_SYM_SHRINKS = (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82)


def _sym_scales(group: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
    """bf16-rounded symmetric scales `[..., N]` of `group` `[..., N, GROUP]`
    minimizing the `importance`-weighted (`[..., GROUP]`, e.g. the Hessian
    diagonal) squared rounding error over a small grid."""
    amax = group.abs().amax(dim=-1).clamp(min=1e-8)
    weights = importance.unsqueeze(-2)
    best = best_err = None
    for divisor in _SYM_DIVISORS:
        for shrink in _SYM_SHRINKS:
            scale = (amax * (shrink / divisor)).bfloat16().float().clamp(min=1e-8)
            q = torch.clamp(torch.round(group / scale.unsqueeze(-1)), -8, 7)
            err = ((group - q * scale.unsqueeze(-1)).square() * weights).sum(-1)
            if best is None:
                best, best_err = scale, err
            else:
                better = err < best_err
                best = torch.where(better, scale, best)
                best_err = torch.where(better, err, best_err)
    assert best is not None
    return best


def prepare_hessian(
    hessian: torch.Tensor, damp: float = _DAMP
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Damped inverse-Hessian Cholesky factors `[..., K, K]` (upper), the
    damped diagonal `[..., K]` and the dead-input mask `[..., K]`."""
    h = hessian.float().clone()
    diag = torch.diagonal(h, dim1=-2, dim2=-1)
    dead = diag == 0
    diag.masked_fill_(dead, 1.0)
    diag.add_(damp * diag.mean(dim=-1, keepdim=True))
    hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True
    )
    return hinv, diag.clone(), dead


@torch.no_grad()
def gptq_w4_sym(
    weight: torch.Tensor, hessian: torch.Tensor, damp: float = _DAMP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric (`uint4b8`) GPTQ of `weight` `[..., N, K]` for the input
    Hessians `hessian` `[..., K, K]` (leading dims batch independent
    matrices): codes `[..., N, K]` (`q + 8`, 0..15) and positive bf16 scales
    `[..., N, K / GROUP]`, with `w ~= (code - 8) * scale`. Groups of 128
    input columns, no activation reordering; each group's scale is searched
    on its error-updated weights."""
    squeeze = weight.dim() == 2
    if squeeze:
        weight, hessian = weight.unsqueeze(0), hessian.unsqueeze(0)
    b, n, k = weight.shape
    assert k % GROUP == 0 and _BLOCK == GROUP
    w = weight.float().clone()
    hinv, diag, dead = prepare_hessian(hessian.to(w.device), damp)
    w.masked_fill_(dead.unsqueeze(1), 0.0)
    codes = torch.empty((b, n, k), dtype=torch.uint8, device=w.device)
    scales = torch.empty((b, n, k // GROUP), dtype=torch.float32, device=w.device)
    for i1 in range(0, k, _BLOCK):
        i2 = i1 + _BLOCK
        g = i1 // GROUP
        w1 = w[:, :, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[:, i1:i2, i1:i2]
        scale = _sym_scales(w1, diag[:, i1:i2])
        scales[:, :, g] = scale
        for i in range(_BLOCK):
            q = torch.clamp(torch.round(w1[:, :, i] / scale), -8, 7)
            codes[:, :, i1 + i] = (q + 8).to(torch.uint8)
            err = (w1[:, :, i] - q * scale) / hinv1[:, i, i].unsqueeze(1)
            w1[:, :, i:] -= err.unsqueeze(2) * hinv1[:, i, i:].unsqueeze(1)
            err1[:, :, i] = err
        if i2 < k:
            w[:, :, i2:] -= torch.bmm(err1, hinv[:, i1:i2, i2:])
    out = codes, scales.bfloat16()
    if squeeze:
        return out[0][0], out[1][0]
    return out


def dequantize_w4_sym(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The fp32 `[..., N, K]` weight of `gptq_w4_sym` codes and scales."""
    *lead, n, k = codes.shape
    w = codes.float().view(*lead, n, k // GROUP, GROUP) - 8
    return (w * scales.float().unsqueeze(-1)).view(*lead, n, k)


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
