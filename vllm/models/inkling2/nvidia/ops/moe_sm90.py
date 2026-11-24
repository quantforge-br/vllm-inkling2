# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Hopper: Humming expert GEMMs (bf16
activations, NVFP4 / MXFP8 weights) in place of the TRT-LLM-gen ones. The
compressed-tensors experts (`moe_ct`) run them on Blackwell too.

Rows stay in the flat (token * topk + slot) order: the gate/up GEMM
gathers each row's token, the down GEMM reads and writes row by row, and
`ops.moe.combine` sums them with the identity `src2dst` (-1: not local).
"""

import functools
import json
import os
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .activation import _situ
from .linear import k_group_partials
from .moe import _load_bias, _local_experts, _select_experts

if TYPE_CHECKING:
    from vllm.utils.humming import HummingInputSchema

# Row block size when no Humming tuning range covers the row count.
_DEFAULT_BLOCK_M = 64
_SM8X_SMALL_BATCH_TUNING = {
    "block_shape": [8, 64, 128],
    "warp_shape": [8, 64, 64],
    "num_ctas_per_sm": 3,
}


@functools.cache
def _fuse_verify_alignment():
    return os.getenv("VLLM_INKLING2_FUSED_SELECT_ALIGN") == "1"


@triton.jit
def _select_kernel(
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
    SortedIds=None,
    BlockExperts=None,
    NumPadded=None,
    ALIGN: tl.constexpr = 0,
):
    """Local expert ids (-1: not local), routing weights and identity
    `src2dst` of the flat rows of one token. With `ALIGN` (a single token),
    also the `moe_align_block_size` layout of its rows for `ALIGN`-row
    blocks: one block per local expert in expert order, the row first and
    the padding row `TOPK` after it."""
    token = tl.program_id(0) + tl.arange(0, 1)
    bias = _load_bias(Bias, E, E_POW2)
    ids, weights = _select_experts(
        Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, SORT
    )
    padding = None
    if IsPadding is not None:
        padding = tl.load(IsPadding + token)
    local = _local_experts(
        ids, tl.load(ExpertBounds), tl.load(ExpertBounds + 1), padding
    )
    rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    tl.store(Ids + rows, local)
    tl.store(Weights + rows, weights)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))
    if ALIGN > 0:
        valid = local >= 0
        # Block of each row: its expert's rank among the local ones.
        rank = tl.sum(
            (valid[:, None, :] & (local[:, None, :] < local[:, :, None])).to(tl.int32),
            2,
        )
        num_valid = tl.sum(valid.to(tl.int32), 1)
        slots = tl.arange(0, TOPK)
        hit = valid[:, None, :] & (rank[:, None, :] == slots[None, :, None])
        tl.store(
            BlockExperts + slots[None, :],
            tl.where(
                slots[None, :] < num_valid[:, None],
                tl.sum(tl.where(hit, local[:, None, :], 0), 2),
                -1,
            ),
        )
        positions = tl.arange(0, TOPK * ALIGN)
        first = valid[:, None, :] & (
            rank[:, None, :] * ALIGN == positions[None, :, None]
        )
        row = tl.sum(tl.where(first, slots[None, None, :], 0), 2)
        is_row = tl.sum(first.to(tl.int32), 2) > 0
        tl.store(SortedIds + positions[None, :], tl.where(is_row, row, TOPK))
        tl.store(NumPadded + tl.arange(0, 1), num_valid * ALIGN)


# Up to this many flat rows, `_align_rows_kernel` lays the rows out.
ALIGN_MAX_ROWS = 128


@triton.jit
def _align_rows_kernel(
    Ids,
    SortedIds,
    BlockExperts,
    NumPadded,
    rows,
    R: tl.constexpr,
    B: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """`moe_align_block_size` of up to `R` flat rows (`Ids`: local expert or
    -1) for `B`-row blocks, in one program: the rows of each expert in row
    order, padded with `rows` to whole blocks, experts in ascending order."""
    index = tl.arange(0, R)
    ids = tl.load(Ids + index, index < rows, other=-1)
    _align_local_ids(ids, SortedIds, BlockExperts, NumPadded, rows, R, B, MAX_BLOCKS)


@triton.jit
def _align_local_ids(
    ids,
    SortedIds,
    BlockExperts,
    NumPadded,
    rows,
    R: tl.constexpr,
    B: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    index = tl.arange(0, R)
    valid = ids >= 0
    same = valid[:, None] & valid[None, :] & (ids[:, None] == ids[None, :])
    earlier = index[None, :] < index[:, None]
    before = tl.sum((same & earlier).to(tl.int32), 1)
    count = tl.sum(same.to(tl.int32), 1)
    first = valid & (before == 0)
    blocks = tl.where(first, (count + B - 1) // B, 0)
    # Blocks of the experts below each row's expert.
    lower = valid[:, None] & first[None, :] & (ids[None, :] < ids[:, None])
    offset = tl.sum(tl.where(lower, blocks[None, :], 0), 1)
    slots = tl.arange(0, R * B)
    tl.store(SortedIds + slots, tl.full((R * B,), rows, tl.int32), slots < rows * B)
    tl.store(BlockExperts + index, tl.full((R,), -1, tl.int32), index < rows)
    # The padding is in place before any thread scatters its rows over it.
    tl.debug_barrier()
    tl.store(SortedIds + offset * B + before, index, valid)
    for b in tl.static_range(MAX_BLOCKS):
        tl.store(BlockExperts + offset + b, ids, first & (b < blocks))
    tl.store(NumPadded + tl.arange(0, 1), tl.sum(blocks) * B + tl.arange(0, 1) * 0)


@triton.jit
def _select_align_verify_kernel(
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    SortedIds,
    BlockExperts,
    NumPadded,
    T: tl.constexpr,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    ALIGN: tl.constexpr,
    TILE: tl.constexpr,
):
    token = tl.arange(0, TILE)
    # The helper loads every row passed to it. Duplicate a valid row in the
    # padded lanes, then exclude those lanes from every store and alignment.
    source = tl.minimum(token, T - 1)
    bias = _load_bias(Bias, E, E_POW2)
    ids, weights = _select_experts(
        Partials, bias, source, T, route_scale, E, E_POW2, GROUPS, TOPK, True
    )
    padding = token >= T
    if IsPadding is not None:
        padding = padding | tl.load(IsPadding + source)
    local = _local_experts(
        ids, tl.load(ExpertBounds), tl.load(ExpertBounds + 1), padding
    )
    rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    valid = token[:, None] < T
    tl.store(Ids + rows, local, valid)
    tl.store(Weights + rows, weights, valid)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1), valid)
    _align_local_ids(
        tl.reshape(local, (TILE * TOPK,)),
        SortedIds,
        BlockExperts,
        NumPadded,
        T * TOPK,
        TILE * TOPK,
        ALIGN,
        tl.cdiv(T, ALIGN),
    )


def align_rows(
    ids: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`moe_align_block_size(ids, block_size, ..., ignore_invalid_experts=True)`
    for at most `ALIGN_MAX_ROWS` flat rows of local expert ids (-1: none)."""
    tokens, topk = ids.shape
    rows = tokens * topk
    assert 0 < rows <= ALIGN_MAX_ROWS and block_size & (block_size - 1) == 0
    assert ids.is_contiguous() and ids.dtype == torch.int32
    sorted_ids = torch.empty(rows * block_size, dtype=torch.int32, device=ids.device)
    block_experts = torch.empty(rows, dtype=torch.int32, device=ids.device)
    num_padded = torch.empty(1, dtype=torch.int32, device=ids.device)
    _align_rows_kernel[(1,)](
        ids,
        sorted_ids,
        block_experts,
        num_padded,
        rows,
        R=triton.next_power_of_2(rows),
        B=block_size,
        # A token routes to an expert once: at most `tokens` rows per expert.
        MAX_BLOCKS=triton.cdiv(tokens, block_size),
        # 3.3 us against 4.6 us with two warps (40 rows, RTX 3090).
        num_warps=8,
    )
    return sorted_ids, block_experts, num_padded


def select_experts(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    align_block: int = 0,
) -> tuple[torch.Tensor, ...]:
    """The router and top-k of `ops.moe.route`, without its layout.

    Returns the local expert id of each token's slots `[T, topk]` (-1: not
    one of the local experts `expert_bounds`, or a padding token), their
    routing weights `[T * topk]` and the identity `src2dst` `[T * topk]`
    of the local rows (-1 elsewhere). With `align_block` (one token; see
    `HummingExperts.decode_align_block`), a fourth element holds the
    `moe_align_block_size` outputs for that block size, from the same launch.
    """
    tokens = x.shape[0]
    experts = gate_weight.shape[0]
    partials = k_group_partials(x, gate_weight)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens * topk, dtype=torch.float32, device=x.device)
    src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=x.device)
    alignment = None
    if align_block:
        assert 1 <= tokens <= 8 and align_block & (align_block - 1) == 0
        rows = tokens * topk
        alignment = (
            torch.empty(rows * align_block, dtype=torch.int32, device=x.device),
            torch.empty(rows, dtype=torch.int32, device=x.device),
            torch.empty(1, dtype=torch.int32, device=x.device),
        )
        if tokens > 1:
            _select_align_verify_kernel[(1,)](
                partials,
                bias,
                expert_bounds,
                is_padding,
                ids,
                weights,
                src2dst,
                *alignment,
                tokens,
                route_scale,
                experts,
                triton.next_power_of_2(experts),
                partials.shape[0],
                topk,
                align_block,
                triton.next_power_of_2(tokens),
                num_warps=8,
            )
            return ids, weights, src2dst, alignment
    _select_kernel[(tokens,)](
        partials,
        bias,
        expert_bounds,
        is_padding,
        ids,
        weights,
        src2dst,
        tokens,
        route_scale,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        SortedIds=alignment[0] if alignment else None,
        BlockExperts=alignment[1] if alignment else None,
        NumPadded=alignment[2] if alignment else None,
        ALIGN=align_block,
        num_warps=4 if tokens <= 256 else 1,
    )
    if alignment is not None:
        return ids, weights, src2dst, alignment
    return ids, weights, src2dst


@triton.jit
def _swiglu_kernel(
    GateUp,
    Weights,
    Src2Dst,
    Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    """bf16 softcapped SwiGLU of a local row times its routing weight."""
    row = tl.program_id(0)
    if tl.load(Src2Dst + row) < 0:
        return
    start = tl.program_id(1) * BLOCK
    gamma = tl.load(Weights + row)
    values = _situ(GateUp, row, start, gamma, N, BLOCK, GATE_CAP, UP_CAP)
    tl.store(Out + row * N + start + tl.arange(0, BLOCK), values.to(tl.bfloat16))


def swiglu(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """`activation.swiglu_quantize`'s SiTU of the interleaved (gate, up)
    columns times the routing weights, in bf16, for the local rows (the
    others are left unset)."""
    rows, n = gate_up.shape[0], gate_up.shape[1] // 2
    block = 256 if n % 256 == 0 else 128
    assert n % block == 0
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=gate_up.device)
    _swiglu_kernel[(rows, n // block)](
        gate_up,
        weights,
        src2dst,
        out,
        N=n,
        BLOCK=block,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        num_warps=2,
    )
    return out


class HummingExperts:
    """Humming tensors and kernel configs of the two expert projections.

    Args:
        weights: Per projection (`w13`, `w2`), the row-major checkpoint
            weights `[E, N, K / pack]`, block scales `[E, N, K / group]`
            and, for NVFP4, the per-expert global scales `[E]` (else None);
            or, for other checkpoint formats (`nvfp4` is then ignored), a
            Humming source weight schema and its tensors by name, with a
            leading expert dim.
        shapes: `(N, K)` of each projection.
        nvfp4: NVFP4 (else MXFP8) weights.
        input_schema: Quantized input contract, or None for BF16 activations.

    """

    def __init__(
        self,
        weights: dict[str, tuple],
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        input_schema: "HummingInputSchema | None" = None,
    ) -> None:
        from vllm.utils.humming import (
            BaseWeightSchema,
            GemmType,
            HummingInputSchema,
            ModeloptMxfp8WeightSchema,
            ModeloptNvfp4WeightSchema,
            get_heuristics_config,
            prepare_and_transform_humming_tensors,
        )

        self.configs = {}
        self.tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.tunings: dict[str, list] = {}
        self.tuning_strs: dict[str, str] = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            source: BaseWeightSchema
            if len(weights[name]) == 2:
                source, tensors = weights[name]
                weight = tensors[source.weight_key]
            else:
                weight, scale, global_scale = weights[name]
                tensors = {"weight": weight, "weight_scale": scale}
                if nvfp4:
                    source = ModeloptNvfp4WeightSchema()
                    tensors["weight_scale_2"] = global_scale
                else:
                    source = ModeloptMxfp8WeightSchema()
            num_experts = weight.shape[0]
            schema, tensors = source.convert_humming(
                tensors=tensors,
                shape_n_stacks=[n],
                shape_k_stacks=[k],
                param_dtype=torch.bfloat16,
                num_experts=num_experts,
            )
            config, tensors = prepare_and_transform_humming_tensors(
                tensors=tensors,
                shape_n=n,
                shape_k=k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema or HummingInputSchema(),
                weight_schema=schema,
                num_experts=num_experts,
                torch_dtype=torch.bfloat16,
                device=weight.device,
            )
            assert config.shape_n == n and config.shape_k == k, (n, k)
            tuning = get_heuristics_config(
                layer_config=config,
                use_f16_accum=False,
                use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
                gemm_type=GemmType.INDEXED,
            )
            # Stream-K adds the K slices into the output with bf16 atomics,
            # whose order (and so the result) changes from run to run.
            for _, _, kernel_config in tuning:
                kernel_config["use_stream_k"] = False
            if not current_platform.has_device_capability(90):
                # SM8x, a few tokens (speculative verification, 16+ local
                # rows): narrower N tiles on three CTAs per SM are 0.7% of a
                # verify step faster than the heuristic's (8, 256, 128) on
                # RTX 3090, with the same accumulation order.
                for lower, _, kernel_config in tuning:
                    if lower == 16:
                        kernel_config.update(_SM8X_SMALL_BATCH_TUNING)
            self.configs[name] = config
            self.tensors[name] = tensors
            self.tunings[name] = tuning
            self.tuning_strs[name] = json.dumps(tuning)
        self.num_experts = num_experts
        self.compute_config = json.dumps(
            {
                "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
                "use_f16_accum": False,
                "gemm_type": GemmType.INDEXED.value,
            }
        )
        self.locks = torch.zeros(2048, dtype=torch.int32, device=weight.device)

    def _block_m(self, name: str, rows: int) -> int:
        for lower, upper, config in self.tunings[name]:
            if lower < rows <= upper:
                return config["block_shape"][0]
        return _DEFAULT_BLOCK_M

    def _gemm(
        self,
        name: str,
        inputs: torch.Tensor,
        rows: int,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        top_k: int,
        valid_rows: int,
    ) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        config = self.configs[name]
        tensors = self.tensors[name]
        out = torch.empty(
            (rows, config.shape_n), dtype=torch.bfloat16, device=inputs.device
        )
        sorted_ids, expert_ids, num_tokens_padded = alignment
        return humming_forward(
            config,
            inputs=inputs,
            weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"),
            zero_point=tensors.get("zero_point"),
            weight_scale_2=tensors.get("weight_scale_2"),
            outputs=out,
            locks=self.locks,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            valid_shape_m=valid_rows,
            compute_config=self.compute_config,
            tuning_config=self.tuning_strs[name],
        )

    def _block_sizes(
        self, rows: int, num_global_experts: int
    ) -> tuple[int, dict[str, int]]:
        # Expected local rows, which pick the kernel configs.
        valid_rows = -(-rows * self.num_experts // num_global_experts)
        return valid_rows, {
            name: self._block_m(name, valid_rows) for name in ("w13", "w2")
        }

    def decode_align_block(
        self, tokens: int, topk: int, num_global_experts: int
    ) -> int:
        """The row block size `select_experts` can align a batch of `tokens`
        for in its own launch (0: none; `moe_align_block_size` runs)."""
        if tokens != 1 and not (
            2 <= tokens <= 8 and topk == 8 and _fuse_verify_alignment()
        ):
            return 0
        _, block_m = self._block_sizes(tokens * topk, num_global_experts)
        sizes = set(block_m.values())
        size = sizes.pop()
        return size if not sizes and size & (size - 1) == 0 else 0

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (`select_experts`); the other rows are left unset. `alignment`
        is that of `select_experts(align_block=decode_align_block(...))`."""
        tokens, topk = ids.shape
        rows = tokens * topk
        valid_rows, block_m = self._block_sizes(rows, num_global_experts)
        if alignment is not None:
            alignments = {size: alignment for size in set(block_m.values())}
            assert len(alignments) == 1
        elif rows <= ALIGN_MAX_ROWS and all(
            size & (size - 1) == 0 for size in block_m.values()
        ):
            # A few tokens (speculative verification): one small launch
            # instead of the two of `moe_align_block_size`.
            alignments = {size: align_rows(ids, size) for size in set(block_m.values())}
        else:
            alignments = {
                size: moe_align_block_size(
                    ids, size, self.num_experts, ignore_invalid_experts=True
                )[:3]
                for size in set(block_m.values())
            }
        gate_up = self._gemm(
            "w13", x, rows, alignments[block_m["w13"]], topk, valid_rows
        )
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm(
            "w2", activated, rows, alignments[block_m["w2"]], 1, valid_rows
        )
