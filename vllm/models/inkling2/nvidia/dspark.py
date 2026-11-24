# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark draft (`mtp.safetensors`): an anchor plus mask block run
through a small dense transformer over DFlare-mixed target-feature K/V, then a
recurrent (`rnn_hidden`) Markov head. Numerics follow the reference
implementation's draft (flashinfer RMSNorms, bf16 logits)."""

import os
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax

from ..configs import Inkling2DSparkConfig
from ..speculator_config import Inkling2DSparkSpeculatorConfig
from .ops import use_portable_kernels
from .ops.attention import block_decode_attention, plan_decode_attention
from .ops.dspark import (
    add_softcap,
    markov_gate,
    mix_rmsnorm,
    qk_norm_rope_,
    rank_combine,
    rope_,
    shard_reduce,
    silu_and_mul,
)
from .ops.int4_linear import is_packed_int4, unpack_int4_rows
from .ops.int8_linear import (
    head_bits,
    int8_dense_enabled,
    int8_linear,
    is_packed_int8,
    markov_m1_enabled,
    markov_m1_int8,
    pack_int8_rows,
    unpack_int8_rows,
)
from .ops.linear import linear
from .ops.marlin_linear import prepare_marlin_w8
from .ops.p2p_draft_norm import draft_norm_workspace, p2p_draft_norm


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.eps = eps
        self.p2p_workspace = None

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        from flashinfer.norm import fused_add_rmsnorm, rmsnorm

        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
        # In place: residual += x; x = norm(residual), from the unrounded sum.
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x


def all_reduce_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, norm: _RMSNorm
) -> torch.Tensor:
    """In place: `residual += all_reduce(x); x = norm(residual)`.

    Batches that the TP group's FlashInfer MNNVL all-reduce takes run its
    fused all-reduce + residual add + RMSNorm on the same workspace (the
    same rank-order fp32 sum, rounded once to bf16); the norm then reads the
    bf16 residual. Others all-reduce, then add and normalize.
    """
    if norm.p2p_workspace is not None and norm.p2p_workspace.usable(*x.shape):
        return p2p_draft_norm(x, residual, norm.weight, norm.eps, norm.p2p_workspace)
    fi_ar = getattr(get_tp_group().device_communicator, "fi_ar_comm", None)
    if fi_ar is None or not fi_ar.should_use_fi_ar(x):
        return norm(tensor_model_parallel_all_reduce(x), residual)
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        flashinfer_comm,
        flashinfer_trtllm_fused_allreduce_norm,
    )
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_workspace,
    )

    workspace = get_fi_ar_workspace(
        fi_ar.world_size,
        fi_ar.rank,
        fi_ar.max_num_tokens,
        x.shape[1],
        x.dtype,
        fi_ar.group,
    )
    assert flashinfer_comm is not None and workspace is not None
    assert workspace.backend == "mnnvl"
    assert x.dtype == residual.dtype == norm.weight.dtype == torch.bfloat16
    assert x.shape == residual.shape and residual.is_contiguous()
    flashinfer_trtllm_fused_allreduce_norm(
        allreduce_in=x,
        residual=residual,
        rms_gamma=norm.weight,
        rms_eps=norm.eps,
        world_size=fi_ar.world_size,
        launch_with_pdl=True,
        fp32_acc=True,
        max_token_num=fi_ar.max_num_tokens,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
    )
    return x


class Inkling2DraftSelfAttention(nn.Module):
    """Bidirectional block attention over context K/V (from target features)
    and the block itself; local layers see `sliding_window_size` positions
    before the block start."""

    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= config.num_key_value_heads
        hidden, self.head_dim = config.hidden_size, config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps
        self.block = config.draft_block_size
        self.is_local = is_local
        self.qkv_proj = QKVParallelLinear(
            hidden,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.kv_proj_ctx = MergedColumnParallelLinear(
            hidden,
            [config.num_key_value_heads * self.head_dim] * 2,
            bias=False,
            prefix=f"{prefix}.kv_proj_ctx",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            hidden,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm_ctx = _RMSNorm(self.head_dim, self.eps)
        # Owns the paged KV cache only; the same spec as the target's layers
        # (the scheduler keeps the last sliding_window_size positions before
        # the block, which is what the draft reads).
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=config.sliding_window_size if is_local else None,
            prefix=f"{prefix}.attn",
            attn_backend=FlashAttentionBackend,
        )
        # Keys a block sees: itself and, if local, the window before it.
        self.window = config.sliding_window_size + self.block if is_local else 0
        vllm_config = get_current_vllm_config()
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.register_buffer(
            "cu_seqlens_q",
            torch.arange(max_num_seqs + 1, dtype=torch.int32) * self.block,
            persistent=False,
        )

    def _metadata(self) -> FlashAttentionMetadata | None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or self.attn.kv_cache.numel() == 0:
            return None
        return cast(FlashAttentionMetadata, attn_metadata[self.attn.layer_name])

    def _write_kv(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        impl = cast(FlashAttentionImpl, self.attn.impl)
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        qkv = linear(
            x, getattr(self.qkv_proj.weight, "marlin_w8", self.qkv_proj.weight)
        )
        qk_norm_rope_(
            qkv,
            positions,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            self.num_heads,
            self.num_kv_heads,
            self.eps,
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        out = q.new_zeros((q.shape[0], self.q_size))
        md = self._metadata()
        if md is not None:
            self._write_kv(
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                md.slot_mapping[: q.shape[0]],
            )
            key_cache, value_cache = self.attn.kv_cache.transpose(1, 2).split(
                self.head_dim, dim=-1
            )
            num_reqs = min(q.shape[0] // self.block, md.seq_lens.shape[0])
            num_tokens = num_reqs * self.block
            # The plan's 0 (FA4, for very large target batches) is one split
            # here.
            num_splits = max(
                1,
                plan_decode_attention(
                    self.is_local,
                    num_reqs,
                    self.block,
                    self.num_heads,
                    self.num_kv_heads,
                    self.window,
                ),
            )
            block_decode_attention(
                q[:num_tokens].view(num_tokens, self.num_heads, self.head_dim),
                canonicalize_singleton_dim_strides(key_cache),
                canonicalize_singleton_dim_strides(value_cache),
                out[:num_tokens],
                block_table=md.block_table[:num_reqs],
                cache_seqlens=md.seq_lens[:num_reqs],
                cu_seqlens_q=self.cu_seqlens_q[: num_reqs + 1],
                block=self.block,
                window=self.window,
                num_splits=num_splits,
            )
        return linear(out, getattr(self.o_proj.weight, "marlin_w8", self.o_proj.weight))

    def store_context_kv(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        k, v = linear(
            context,
            getattr(self.kv_proj_ctx.weight, "marlin_w8", self.kv_proj_ctx.weight),
        ).split([self.kv_size, self.kv_size], -1)
        k = self.k_norm_ctx(k.reshape(-1, self.head_dim))
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        rope_(k, positions, cos_sin_cache)
        if slot_mapping is not None and self.attn.kv_cache.numel() > 0:
            self._write_kv(
                k, v.view(-1, self.num_kv_heads, self.head_dim), slot_mapping
            )


class Inkling2DraftMLP(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = silu_and_mul(
            linear(
                x,
                getattr(
                    self.gate_up_proj.weight, "marlin_w8", self.gate_up_proj.weight
                ),
            )
        )
        return linear(
            x, getattr(self.down_proj.weight, "marlin_w8", self.down_proj.weight)
        )


class Inkling2DraftLayer(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _RMSNorm(hidden, eps)
        self.post_attention_layernorm = _RMSNorm(hidden, eps)
        self.dflare_norm = _RMSNorm(hidden, eps)
        self.self_attn = Inkling2DraftSelfAttention(
            config, is_local, f"{prefix}.self_attn"
        )
        self.mlp = Inkling2DraftMLP(config, f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns this rank's partial MLP output and the residual."""
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x = all_reduce_add_rmsnorm(x, residual, self.input_layernorm)
        x = self.self_attn(positions, x, cos_sin_cache)
        x = all_reduce_add_rmsnorm(x, residual, self.post_attention_layernorm)
        return self.mlp(x), residual


class Inkling2MarkovHead(nn.Module):
    """Recurrent (`rnn_hidden`) correction of the draft's base logits.

    Its rank-`r` latent goes through `hidden_out` and the shared LM head,
    folded once into a `(vocab, r)` projection. The `[hidden, f_rnn]`
    columns of `rnn_proj` (and its bias) are applied to all draft steps at
    once in fp32; each step adds the `[state, previous]` columns and rounds.
    """

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(
            config.padded_vocab_size,
            rank,
            _weight=torch.empty(config.padded_vocab_size, rank),
            _freeze=True,
        )
        self.rnn_proj = ReplicatedLinear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = ReplicatedLinear(rank, hidden, bias=True)
        self.dflare_norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.fold_weight: torch.Tensor | None = None
        self.fold_bias: torch.Tensor | None = None
        # W8A16 copy of `fold_weight` for the decode GEMVs (int8 dense mode).
        self.fold_packed: torch.Tensor | None = None
        self.rnn_bias: torch.Tensor | None = None

    @torch.no_grad()
    def fold(self, lm_head_weight: torch.Tensor) -> None:
        """Fold `hidden_out` into the (TP-local) LM head rows in fp64. Later
        calls refill the same tensors, which CUDA graphs keep pointing at."""
        projection = self.hidden_out.weight.double()
        bias = self.hidden_out.bias.double()
        num_rows = lm_head_weight.shape[0]
        if self.fold_weight is None or self.fold_bias is None:
            # Not the head's dtype: a packed W8A16 / W4A16 head is folded
            # from the weights it stands for.
            dense = self.hidden_out.weight
            self.fold_weight = dense.new_empty((num_rows, self.markov_rank))
            self.fold_bias = dense.new_empty((num_rows,))
        for start in range(0, num_rows, 4096):
            rows = slice(start, start + 4096)
            head = lm_head_weight[rows]
            if is_packed_int4(head):
                head = unpack_int4_rows(head, projection.shape[0])
            elif is_packed_int8(head):
                if hasattr(lm_head_weight, "marlin_row_scales"):
                    head = (
                        head[:, : projection.shape[0]].float()
                        * lm_head_weight.marlin_row_scales[rows]
                    )
                else:
                    head = unpack_int8_rows(head)
            head = head.double()
            self.fold_weight[rows].copy_(head @ projection)
            self.fold_bias[rows].copy_(head @ bias)
        # With a bf16 head (`VLLM_INKLING2_HEAD_BITS=16`) the draft stays bf16 too.
        if int8_dense_enabled() and head_bits() < 16 and use_portable_kernels():
            packed = pack_int8_rows(self.fold_weight)
            if self.fold_packed is None:
                self.fold_packed = packed
            else:
                self.fold_packed.copy_(packed)

    def context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """fp32 `rnn_proj` bias plus its `[hidden, f_rnn]` term for the
        `(req, step)`-ordered `hidden`; returns `(step, req, 3 * r)`."""
        assert self.rnn_bias is not None
        num_reqs, width = f_rnn.shape
        hidden = hidden.view(num_reqs, -1, width).transpose(0, 1)
        inputs = torch.cat([hidden, f_rnn.expand_as(hidden)], dim=-1)
        weight = self.rnn_proj.weight[:, 2 * self.markov_rank :]
        out = torch.addmm(
            self.rnn_bias, inputs.flatten(0, 1), weight.T, out_dtype=torch.float32
        )
        return out.view(hidden.shape[0], num_reqs, -1)

    def step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step; returns the new state and the TP-local bias."""
        assert self.fold_weight is not None and self.fold_bias is not None
        inputs = torch.cat([state, self.markov_w1(prev_tokens)], dim=-1)
        weight = self.rnn_proj.weight[:, : 2 * self.markov_rank]
        raw = torch.addmm(context, inputs, weight.T, out_dtype=torch.float32)
        state, latent = markov_gate(raw, state)
        if self.fold_packed is not None and latent.shape[0] <= 16:
            if latent.shape == (1, 256) and markov_m1_enabled():
                logits = markov_m1_int8(latent, self.fold_packed)
            else:
                logits = int8_linear(latent, self.fold_packed)
            return state, logits.add_(self.fold_bias)
        return state, torch.addmm(self.fold_bias, latent, self.fold_weight.T)


class Inkling2DSparkDecoder(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, max_model_len: int, prefix: str
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(
                config, i in config.local_layer_ids, f"{prefix}.layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.markov_head = Inkling2MarkovHead(config)
        self.fuse_weight = nn.Parameter(
            torch.zeros(config.num_hidden_layers + 1, len(config.target_layer_ids)),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self.register_buffer("fuse_alpha", torch.empty_like(self.fuse_weight))
        # fp32 [cos | sin] table, read directly by the draft's RoPE kernels.
        self.cos_sin_cache = get_rope(
            config.head_dim,
            max_position=max_model_len,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            dtype=torch.float32,
        ).cos_sin_cache
        # Shared with the target model (see Inkling2DSparkModel.attach_target).
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None

    def mix(self, features: torch.Tensor, row: int, norm: _RMSNorm) -> torch.Tensor:
        targets = features.view(features.shape[0], -1, self.config.hidden_size)
        return mix_rmsnorm(targets, self.fuse_alpha[row], norm.weight, norm.eps)

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normalized target embedding of each block's anchor, then masks."""
        assert self.embed_tokens is not None and self.embed_norm is not None
        block = self.config.draft_block_size
        ids = input_ids.view(-1, block)
        x = self.mask_embed.expand(ids.shape[0], block, -1).clone()
        x[:, 0] = self.embed_norm(self.embed_tokens(ids[:, 0].contiguous()))
        return x.view(-1, self.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_block(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(positions, x, residual, self.cos_sin_cache)
        assert residual is not None
        return all_reduce_add_rmsnorm(x, residual, self.norm)


class Inkling2DSparkModel(nn.Module):
    """DSpark draft for Inkling2; plugged in by `Inkling2DSparkSpeculator`."""

    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = Inkling2DSparkSpeculatorConfig
    allow_patterns_overrides = ["mtp.safetensors"]
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    _stacked = (
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
        (".k_proj_ctx.", ".kv_proj_ctx.", 0),
        (".v_proj_ctx.", ".kv_proj_ctx.", 1),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        assert isinstance(config, Inkling2DSparkConfig)
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        if num_spec != config.draft_block_size - 1:
            raise ValueError(
                "Inkling2 DSpark requires num_speculative_tokens = "
                f"{config.draft_block_size - 1} (the checkpoint's block size - 1)"
            )
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(
                "Inkling2 DSpark drafts up to max_position_embeddings="
                f"{config.max_position_embeddings} positions; set --max-model-len "
                "to at most that"
            )
        self.config = config
        # Reduce each rank's vocabulary shard before the exchange instead of
        # all-gathering the full logits at every draft position.
        self.sharded_argmax = (
            get_tensor_model_parallel_world_size() > 1
            and os.environ.get("VLLM_INKLING2_DSPARK_SHARDED_ARGMAX", "1") == "1"
        )
        self.model = Inkling2DSparkDecoder(
            config, vllm_config.model_config.max_model_len, f"{prefix}model"
        )
        self.lm_head: nn.Module | None = None

    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        """Finish sharing the target's normalized embeddings and LM head (its
        `embed_tokens` and `lm_head` are aliased by `load_dspark_model`)
        and derive the weights that depend on them."""
        assert self.lm_head is not None
        weight = self.lm_head.weight
        assert (
            weight.dtype == torch.bfloat16
            or is_packed_int8(weight)
            or is_packed_int4(weight)
        )
        self.model.embed_norm = target_model.model.embed_norm
        norm_workspace = draft_norm_workspace(self.config.hidden_size)
        for layer in self.model.layers:
            layer.input_layernorm.p2p_workspace = norm_workspace
            layer.post_attention_layernorm.p2p_workspace = norm_workspace
        self.model.norm.p2p_workspace = norm_workspace
        head = self.model.markov_head
        head.rnn_bias = torch.empty_like(head.rnn_proj.bias, dtype=torch.float32)
        self.refresh_derived_weights()

    @torch.no_grad()
    def refresh_derived_weights(self) -> None:
        """Re-derive, in place, the weights computed from the target's LM head
        and the draft's own weights; called again after the target's weights
        are replaced (reload, RL weight transfer)."""
        assert self.lm_head is not None
        head = self.model.markov_head
        assert head.rnn_bias is not None
        if os.getenv("VLLM_INKLING2_MARLIN_W8_DRAFT") == "1":
            for layer in self.model.layers:
                for proj in (
                    layer.self_attn.qkv_proj,
                    layer.self_attn.kv_proj_ctx,
                    layer.self_attn.o_proj,
                    layer.mlp.gate_up_proj,
                    layer.mlp.down_proj,
                ):
                    prepare_marlin_w8(
                        proj.weight,
                        pack_int8_rows(proj.weight, scale_dtype=torch.bfloat16),
                    )
        head.fold(self.lm_head.weight)
        head.rnn_bias.copy_(head.rnn_proj.bias)
        self.model.fuse_alpha.copy_(
            torch.softmax(self.model.fuse_weight.float(), dim=-1)
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Each draft layer mixes the raw target features itself.
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        for i, layer in enumerate(self.model.layers):
            context = self.model.mix(context_states, i, layer.dflare_norm)
            layer.self_attn.store_context_kv(
                context,
                context_positions,
                self.model.cos_sin_cache,
                None if context_slot_mappings is None else context_slot_mappings[i],
            )

    def compute_f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        """Recurrent-head context from the last committed token's features."""
        return self.model.mix(features, -1, self.model.markov_head.dflare_norm)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """TP-local bf16 base logits (bf16 GEMM, as the reference)."""
        assert self.lm_head is not None
        return linear(hidden_states, self.lm_head.weight)

    def markov_step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the recurrent head; return the state and TP-local bias."""
        return self.model.markov_head.step(state, prev_tokens, context)

    def markov_logits(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Full-vocab softcapped logits."""
        logits = add_softcap(base_logits, bias, self.config.final_logit_softcapping)
        logits = tensor_model_parallel_all_gather(logits)
        return logits[:, : self.config.vocab_size]

    def _sharded_argmax_confidence(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`markov_argmax_confidence` without gathering the vocabulary: each
        rank reduces its shard to (max, index, logsumexp) and one small
        all-reduce exchanges them (ties keep the first index, as
        `greedy_argmax`)."""
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        shard = base_logits.shape[1]
        start = rank * shard
        packed = shard_reduce(
            base_logits,
            bias,
            self.config.final_logit_softcapping,
            start,
            self.config.vocab_size - start,
            rank,
            tp,
        )
        return rank_combine(tensor_model_parallel_all_reduce(packed))

    def markov_argmax(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """`markov_logits(...).argmax(-1)`, NaN as -inf."""
        if self.sharded_argmax:
            return self._sharded_argmax_confidence(base_logits, bias)[0]
        return greedy_argmax(self.markov_logits(base_logits, bias))

    def markov_argmax_confidence(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`markov_argmax` and the draft's fp32 probability of that token."""
        if self.sharded_argmax:
            return self._sharded_argmax_confidence(base_logits, bias)
        logits = self.markov_logits(base_logits, bias)
        tokens = greedy_argmax(logits)
        logits = logits.float()
        top = logits.gather(1, tokens.view(-1, 1)).squeeze(1)
        return tokens, torch.exp(top - torch.logsumexp(logits, dim=-1))

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            name, shard = f"model.{name}", None
            for old, new, shard_id in self._stacked:
                if old in name:
                    name, shard = name.replace(old, new), shard_id
                    break
            param = params[name]
            if weight.dtype != param.dtype:
                raise ValueError(f"{name}: expected {param.dtype}, got {weight.dtype}")
            if shard is None:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            else:
                param.weight_loader(param, weight, shard)
            loaded.add(name)
        return loaded
