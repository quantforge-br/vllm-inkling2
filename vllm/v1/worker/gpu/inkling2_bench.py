# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, single-stream Inkling 2 measurements for SM86 development."""

import json
import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch


@triton.jit
def _inkling2_target_started(marker):
    tl.store(marker, 1)


@triton.jit
def _inkling2_target_finished(marker):
    tl.store(marker, 2)


@triton.jit
def _inkling2_step_finished(marker):
    tl.store(marker, 3)


class Inkling2Measurements:
    @classmethod
    def from_env(cls, config, device):
        directory = os.getenv("VLLM_INKLING2_BENCH_DIR")
        if not directory:
            return None
        if (
            config.model_config.hf_config.model_type
            not in ("inkling2", "inkling2_loop")
            or config.scheduler_config.max_num_seqs != 1
            or config.scheduler_config.async_scheduling
            or config.parallel_config.tensor_parallel_size != 2
            or config.parallel_config.pipeline_parallel_size != 1
            or config.parallel_config.data_parallel_size != 1
        ):
            raise ValueError(
                "Inkling 2 measurement requires synchronous single-stream TP2"
            )
        trace = config.profiler_config.profiler == "torch"
        return cls(Path(directory), device, trace)

    def __init__(self, directory: Path, device: torch.device, trace: bool):
        directory.mkdir(parents=True, exist_ok=True)
        self.rank = get_tensor_model_parallel_rank()
        self.output = (directory / f"worker-rank{self.rank}-{os.getpid()}.jsonl").open(
            "x", buffering=1
        )
        self.marker = (
            torch.empty(1, device=device, dtype=torch.int32) if trace else None
        )
        self.times: dict[str, float] = {}
        self.previous_take: tuple[float, float] | None = None
        self.offered: dict[str, int] = {}

    def execute_start(self):
        self.times = {"execute_enter": perf_counter()}
        if self.previous_take:
            self.times.update(zip(("take_enter", "take_exit"), self.previous_take))
            self.previous_take = None

    def stamp(self, name: str):
        self.times[name] = perf_counter()

    def forward_start(self, is_decode: bool):
        if self.marker is not None and is_decode:
            _inkling2_target_started[(1,)](self.marker, num_warps=1)
        self.stamp("forward_enter")

    def forward_end(self, is_decode: bool):
        self.stamp("forward_exit")
        if self.marker is not None and is_decode:
            _inkling2_target_finished[(1,)](self.marker, num_warps=1)

    def step_end(self, batch: "InputBatch", proposed: int):
        if self.marker is not None and not batch.has_prefill:
            _inkling2_step_finished[(1,)](self.marker, num_warps=1)
        times = self.times
        q_len = int(batch.num_scheduled_tokens[0])
        is_decode = not bool(batch.is_prefilling_np[0])

        row = {
            "request_id": batch.req_ids[0],
            "decode": is_decode,
            "q_len": q_len,
            "draft_slots_computed": proposed,
            "offered_for_step": self.offered.pop(batch.req_ids[0], 0),
            "host_times_s": times,
        }

        if self.rank != 0:
            # TP only returns sampled tokens from rank 0. Keep independent
            # step and host timing on rank 1 without synchronizing its copy.
            self.write(row)
            return None

        def record(output: "ModelRunnerOutput"):
            sampled = len(output.sampled_token_ids[0])
            verified = (output.num_verified_tokens_per_req or [0])[0]
            accepted = min(max(0, sampled - 1), verified)
            self.write(dict(row, sampled=sampled, verified=verified, accepted=accepted))

        return record

    def write(self, row):
        self.output.write(json.dumps(row, separators=(",", ":")) + "\n")

    def take_drafts(self, get_draft_tokens):
        started = perf_counter()
        drafts = get_draft_tokens()
        self.previous_take = started, perf_counter()
        if drafts is not None:
            self.offered = {
                req: len(tokens)
                for req, tokens in zip(
                    drafts.req_ids, drafts.draft_token_ids, strict=True
                )
            }
        return drafts
