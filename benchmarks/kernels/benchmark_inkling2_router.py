# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare SM86 routing kernels on the TP2 half-expert shapes."""
# ruff: noqa: B023

import argparse
import itertools
import json
import random

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.models.inkling2.nvidia.ops.linear import _k_group_kernel, k_group_partials
from vllm.models.inkling2.nvidia.ops.moe_sm90 import align_rows, select_experts
from vllm.triton_utils import triton


def bench(fn, rep):
    return 1000 * triton.testing.do_bench_cudagraph(fn, rep=rep)


def k_group(inputs, weight, rep):
    tokens, hidden = inputs.shape
    experts = len(weight)
    ref = k_group_partials(inputs, weight)
    total = ref[0] + ref[1] + ref[2]
    configs = list(
        itertools.product((16, 32, 64), (2, 4), (1, 2, 3), (True, False), (True, False))
    )
    random.Random(11).shuffle(configs)
    for bn, warps, stages, split, prefetch in configs:
        out = torch.empty_like(ref if split else ref[:1])

        def launch():
            _k_group_kernel[(triton.cdiv(experts, bn), 1, len(out))](
                inputs,
                weight,
                out,
                tokens,
                experts,
                hidden,
                16,
                bn,
                split,
                prefetch,
                num_warps=warps,
                num_stages=stages,
                enable_fp_fusion=False,
            )

        launch()
        equal = torch.equal(out, ref) if split else torch.equal(out[0], total)
        print(
            json.dumps(
                dict(
                    T=tokens,
                    BN=bn,
                    warps=warps,
                    stages=stages,
                    split=split,
                    prefetch=prefetch,
                    exact=equal,
                    us=bench(launch, rep),
                )
            ),
            flush=True,
        )


def selection(inputs, weight, rep):
    tokens = len(inputs)
    bias = torch.randn(128, device=inputs.device) * 0.1
    for lo, block in itertools.product((0, 64), (8, 16, 64)):
        bounds = torch.tensor([lo, lo + 64], device=inputs.device, dtype=torch.int32)
        for padded in (False, True):
            padding = (
                torch.arange(tokens, device=inputs.device) % 3 == 0 if padded else None
            )

            def launch(fused):
                values = select_experts(
                    inputs, weight, bias, 8, 1.0, bounds, padding, block if fused else 0
                )
                return values if fused else (*values, align_rows(values[0], block))

            original = launch(False)
            expected = moe_align_block_size(
                original[0], block, 64, ignore_invalid_experts=True
            )
            actual = launch(True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay = launch(True)
            graph.replay()
            n = int(expected[2])
            for result in (actual, replay):
                for a, b in zip(original[:3], result[:3], strict=True):
                    assert torch.equal(a, b)
                for a, b in zip(original[3], result[3], strict=True):
                    assert torch.equal(a, b)
                assert n == int(result[3][2])
                assert torch.equal(result[3][0][:n], expected[0][:n])
                assert torch.equal(
                    result[3][1][: n // block], expected[1][: n // block]
                )
        print(
            json.dumps(
                dict(
                    T=tokens,
                    block=block,
                    rank=lo // 64,
                    exact=True,
                    original_us=bench(lambda: launch(False), rep),
                    fused_us=bench(lambda: launch(True), rep),
                )
            ),
            flush=True,
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("k-group", "selection"))
    parser.add_argument("--rep-ms", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(13)
    weight = (torch.randn(128, 2304, device="cuda") * 0.05).bfloat16()
    for t in range(1, 9):
        inputs = torch.randn(t, 2304, device="cuda").bfloat16()
        (selection if args.experiment == "selection" else k_group)(
            inputs, weight, args.rep_ms
        )


if __name__ == "__main__":
    main()
