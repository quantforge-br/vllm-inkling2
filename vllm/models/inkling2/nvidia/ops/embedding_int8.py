# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 input embedding (`VLLM_INKLING2_EMBED_INT8=1`): each rank keeps its
vocabulary shard as symmetric int8 codes with one bf16 scale per 128 columns
and dequantizes only the gathered rows, in one launch (the bf16 table is the
largest weight left on the GPU after the int4 experts)."""

import functools
import os
from types import MethodType

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.triton_utils import tl, triton

GROUP = 128


@functools.cache
def embed_int8_enabled() -> bool:
    return os.environ.get("VLLM_INKLING2_EMBED_INT8", "0") == "1"


def quantize_embedding_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 codes `[V, H]` and bf16 scales `[V, H / GROUP]` of `weight`, with
    codes rounded against the bf16-rounded scales."""
    v, h = weight.shape
    assert h % GROUP == 0, h
    w = weight.float().view(v, h // GROUP, GROUP)
    amax = w.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).bfloat16()
    codes = torch.round(w / scale.float().unsqueeze(-1)).clamp_(-127, 127)
    return codes.to(torch.int8).view(v, h), scale


@triton.jit
def _gather_kernel(
    ids_ptr,
    codes_ptr,
    scales_ptr,
    out_ptr,
    start,
    end,
    H: tl.constexpr,
    G: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    token = tl.load(ids_ptr + row).to(tl.int64)
    valid = (token >= start) & (token < end)
    local = tl.where(valid, token - start, 0)
    cols = group * G + tl.arange(0, G)
    codes = tl.load(codes_ptr + local * H + cols).to(tl.float32)
    scale = tl.load(scales_ptr + local * (H // G) + group).to(tl.float32)
    out = tl.where(valid, codes * scale, 0.0)
    tl.store(out_ptr + row * H + cols, out.to(tl.bfloat16))


def gather_int8(
    ids: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    """bf16 rows `[T, H]` of the ids inside `[start, end)` (zero elsewhere)."""
    flat = ids.reshape(-1)
    h = codes.shape[1]
    out = torch.empty((flat.shape[0], h), dtype=torch.bfloat16, device=codes.device)
    if flat.numel():
        _gather_kernel[(flat.shape[0], h // GROUP)](
            flat, codes, scales, out, start, end, H=h, G=GROUP, num_warps=1
        )
    return out.view(*ids.shape, h)


def _forward(self: nn.Module, input_: torch.Tensor) -> torch.Tensor:
    indices = self.shard_indices
    out = gather_int8(
        input_,
        self.int8_codes,
        self.int8_scales,
        indices.org_vocab_start_index,
        indices.org_vocab_end_index,
    )
    if get_tensor_model_parallel_world_size() > 1:
        out = tensor_model_parallel_all_reduce(out)
    return out


@torch.no_grad()
def make_embedding_int8(embed: nn.Module) -> None:
    """Turns a vLLM `VocabParallelEmbedding` (no added vocabulary) into the
    int8 form, in place."""
    indices = embed.shard_indices
    if indices.num_added_elements:
        raise ValueError(
            "VLLM_INKLING2_EMBED_INT8 needs an embedding without added vocabulary rows"
        )
    weight = embed.weight
    codes, scales = quantize_embedding_int8(weight.data)
    del embed.weight
    embed.weight = torch.empty(0, device=weight.device)
    embed.register_buffer("int8_codes", codes, persistent=False)
    embed.register_buffer("int8_scales", scales, persistent=False)
    embed.forward = MethodType(_forward, embed)
