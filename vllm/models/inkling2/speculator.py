# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator for the Inkling2 draft: stores the raw target features
(each draft layer mixes them itself) and runs its recurrent Markov head."""

import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None and not self.enable_adaptive_verification
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            len(config.target_layer_ids) * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # combine_hidden_states is the identity: store features eagerly.
        self._aux_staging = None
        # Read by prepare_context_anchor inside the FULL draft graph.
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )
        # Drafts are verified only while the draft's own probability of the
        # whole prefix stays at or above this (0: verify every draft). The
        # scheduler reads the prefix lengths, so it needs synchronous
        # scheduling and greedy drafting.
        self.min_prefix_confidence = float(
            os.environ.get("VLLM_INKLING2_DSPARK_MIN_CONFIDENCE", "0")
        )
        self.adaptive_prefix = self.min_prefix_confidence != 0
        self.draft_prefix_lens: torch.Tensor | None = None
        if self.adaptive_prefix:
            if not 0 < self.min_prefix_confidence <= 1:
                raise ValueError(
                    "VLLM_INKLING2_DSPARK_MIN_CONFIDENCE must be in (0, 1], got "
                    f"{self.min_prefix_confidence}"
                )
            if vllm_config.scheduler_config.async_scheduling:
                # Async scheduling fixes the next step's rows with placeholders
                # before the drafts exist.
                raise ValueError(
                    "VLLM_INKLING2_DSPARK_MIN_CONFIDENCE requires "
                    "--no-async-scheduling"
                )
            assert vllm_config.speculative_config is not None
            if vllm_config.speculative_config.draft_sample_method != "greedy":
                # The confidence branch does not fill the draft logits cache
                # that probabilistic rejection sampling reads.
                raise ValueError(
                    "VLLM_INKLING2_DSPARK_MIN_CONFIDENCE requires greedy drafting"
                )
            self.draft_prefix_lens = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if self.adaptive_prefix and (
            self.draft_logits is not None
            or getattr(self, "_sharded_sequential_comm", None) is not None
        ):
            # Only the replicated greedy `_sample_sequential` below counts
            # the prefix (both are settled by the base class while loading).
            raise ValueError(
                "VLLM_INKLING2_DSPARK_MIN_CONFIDENCE requires replicated greedy "
                "drafting"
            )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Draft base logits come from a bf16 GEMM, unlike the target's fp32 head.
        return self.dtype, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        return model

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        prefix_prob = prefix_len = None
        if self.adaptive_prefix:
            prefix_prob = hidden.new_ones(num_reqs, dtype=torch.float32)
            prefix_len = torch.zeros(num_reqs, dtype=torch.int32, device=hidden.device)
        for i in range(n_spec):
            state, bias = self.model.markov_step(state, prev, context[i])
            if prefix_prob is not None:
                prev, confidence = self.model.markov_argmax_confidence(
                    base_logits[:, i], bias
                )
                # The product only falls, so this counts a prefix.
                prefix_prob = prefix_prob * confidence
                prefix_len = prefix_len + (prefix_prob >= self.min_prefix_confidence)
            elif self.draft_logits is None:
                prev = self.model.markov_argmax(base_logits[:, i], bias)
            else:
                logits = self.model.markov_logits(base_logits[:, i], bias)
                prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev
        if prefix_len is not None:
            assert self.draft_prefix_lens is not None
            self.draft_prefix_lens[:num_reqs] = prefix_len
