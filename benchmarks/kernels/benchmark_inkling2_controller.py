# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM86 controller sweep with fixed-shape eager and CUDA graph parity."""

import argparse
import itertools
import json
import os

import torch

from vllm.models.inkling2.nvidia.ops.loop import _grouped_controller, controller_blend
from vllm.triton_utils import triton


def make_inputs(streams, tokens, shared):
    d, r = 2304, 64

    def random(*shape):
        return torch.randn(shape, device="cuda")

    dest = random(1 if shared else streams, tokens, d) * 5
    if shared:
        dest = dest.expand(streams, -1, -1)
    return (
        dest,
        random(streams, tokens, d) * 3,
        (random(streams, r, 2 * d) * 0.03).bfloat16(),
        (random(streams, 2 * d, r) * 0.05).bfloat16(),
        (1 + random(streams, d) * 0.1).bfloat16(),
        1e-5,
    )


def select(group, prepass):
    os.environ["VLLM_INKLING2_CONTROLLER_TOKEN_GROUP"] = str(group)
    os.environ["VLLM_INKLING2_CONTROLLER_RSTD_PREPASS"] = str(int(prepass))
    _grouped_controller.cache_clear()


def check(inputs, baseline, group, prepass, rep):
    select(group, prepass)
    actual = controller_blend(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = controller_blend(*inputs)
    graph.replay()
    torch.testing.assert_close(actual, replay, rtol=0, atol=0)
    diffs = actual.float() - baseline.float()
    ms = triton.testing.do_bench_cudagraph(lambda: controller_blend(*inputs), rep=rep)
    return {
        "group": group,
        "prepass": prepass,
        "gpu_us": ms * 1000,
        "mismatches": diffs.count_nonzero().item(),
        "max_abs": diffs.abs().max().item(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep-ms", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    designs = [(1, False), *itertools.product((2, 4, 8), (False, True))]
    for tokens, (streams, shared) in itertools.product(
        range(1, 9), [(1, False), (3, False), (3, True)]
    ):
        inputs = make_inputs(streams, tokens, shared)
        select(1, False)
        baseline = controller_blend(*inputs)
        for group, prepass in designs:
            result = check(inputs, baseline, group, prepass, args.rep_ms)
            print(
                json.dumps(
                    dict(
                        result,
                        tokens=tokens,
                        streams=streams,
                        shared=shared,
                        seed=args.seed,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
