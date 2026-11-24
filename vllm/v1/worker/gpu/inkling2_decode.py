# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static row layouts for single-request, synchronous Inkling 2 DSpark decode."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch


@triton.jit
def _prepare_decode(
    Positions, SeqLens, QueryStart, InputIds, Computed, Last, Draft, WIDTH: tl.constexpr
):
    lane = tl.arange(0, 8)
    base = tl.load(Computed)
    tl.store(Positions + lane, base + lane, mask=lane < WIDTH)
    tl.store(SeqLens, base + WIDTH)
    # These addresses also belong to the captured target/draft graphs.
    tl.store(QueryStart, 0)
    tl.store(QueryStart + 1, WIDTH)
    tl.store(InputIds, tl.load(Last))
    if WIDTH > 1:
        draft = tl.load(Draft + lane, mask=lane < WIDTH - 1, other=0)
        tl.store(InputIds + lane + 1, draft, mask=lane < WIDTH - 1)


@dataclass(frozen=True)
class DecodeLayout:
    offsets_cpu: np.ndarray
    offsets: torch.Tensor
    req_rows: torch.Tensor
    local_pos: torch.Tensor
    logits_indices: torch.Tensor

    @staticmethod
    def create(width, device):
        cpu = torch.tensor([0, width], dtype=torch.int32, pin_memory=True)
        offsets = cpu.numpy()
        offsets.flags.writeable = False
        return DecodeLayout(
            offsets,
            cpu.to(device),
            torch.zeros(width, dtype=torch.int64, device=device),
            torch.arange(width, dtype=torch.int32, device=device),
            torch.arange(width, dtype=torch.int64, device=device),
        )

    def prepare(
        self, states, buffers, batch, scheduler_output, padded_tokens, is_padding
    ):
        width = batch.num_tokens
        _prepare_decode[(1,)](
            buffers.positions,
            buffers.seq_lens,
            buffers.query_start_loc,
            buffers.input_ids,
            states.num_computed_tokens.gpu,
            states.last_sampled_tokens,
            states.draft_tokens,
            WIDTH=width,
        )
        upper = states.num_computed_tokens_np[:1].copy()
        upper += width
        return InputBatch(
            req_ids=batch.req_ids,
            num_reqs=1,
            num_reqs_after_padding=1,
            idx_mapping=self.req_rows[:1],
            idx_mapping_np=batch.idx_mapping_np,
            expanded_idx_mapping=self.req_rows,
            expanded_local_pos=self.local_pos,
            num_scheduled_tokens=batch.num_scheduled_tokens,
            num_tokens=width,
            num_tokens_after_padding=padded_tokens,
            num_draft_tokens=width - 1,
            num_draft_tokens_per_req=batch.num_draft_tokens_np,
            num_admitted_draft_tokens=None,
            query_start_loc=buffers.query_start_loc[:2],
            query_start_loc_np=self.offsets_cpu,
            seq_lens=buffers.seq_lens[:1],
            seq_lens_cpu_upper_bound=torch.from_numpy(upper),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=upper - width,
            prefill_len_np=batch.prefill_len_np,
            num_computed_prefill_tokens_np=batch.num_computed_prefill_tokens_np,
            is_prefilling_np=batch.is_prefilling_np,
            has_prefill=False,
            prefill_runs_as_decode_np=None,
            decode_graph_eligible=True,
            max_seq_len_np=batch.max_seq_len_np,
            input_ids=buffers.input_ids[:padded_tokens],
            positions=buffers.positions[:padded_tokens],
            is_padding=is_padding,
            logits_indices=self.logits_indices,
            cu_num_logits=self.offsets,
            cu_num_logits_np=self.offsets_cpu,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=None,
            fast_prefill=None,
            max_query_len=None,
        )
