# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM86 exactness and timing of the multi-query residual kernel."""
# ruff: noqa: B023

import argparse
import itertools
import json
import os

import torch

from vllm.models.inkling2.nvidia.ops.attn_res import _multi_query_mode, attn_res
from vllm.triton_utils import triton


def mode(value):
    os.environ["VLLM_INKLING2_ATTN_RES_MULTI_QUERY"] = str(value)
    _multi_query_mode.cache_clear()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep-ms", type=int, default=5)
    parser.add_argument("--stacks", nargs="+", type=int, default=list(range(1, 13)))
    args = parser.parse_args()
    torch.manual_seed(2)
    for n, q, t in itertools.product(args.stacks, (1, 3, 4), range(1, 9)):
        d = 2304
        stack = torch.randn(n, t, d, device="cuda", dtype=torch.bfloat16)
        last = torch.randn_like(stack[0])
        extra = torch.randn_like(last)
        queries = torch.randn(q, d, device="cuda") * d**-0.5
        weight = torch.randn_like(queries).bfloat16()
        for add, norm, normalized in itertools.product((False, True), repeat=3):

            def run():
                return attn_res(
                    stack,
                    last,
                    extra if add else None,
                    queries,
                    weight if norm else None,
                    1e-5,
                    normalized,
                )

            mode(0)
            expected = run()
            for m in (1, 2) if n <= 2 else (1,):
                mode(m)
                actual = run()
                assert torch.equal(actual, expected), (
                    n,
                    q,
                    t,
                    add,
                    norm,
                    normalized,
                    m,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay = run()
                graph.replay()
                assert torch.equal(replay, expected)
        times = {}
        for m in (0, 1, 2) if n <= 2 else (0, 1):
            mode(m)
            times[m] = 1000 * triton.testing.do_bench_cudagraph(run, rep=args.rep_ms)
        print(
            json.dumps(dict(n=n, q=q, t=t, variants_us=times, exact=True)), flush=True
        )


if __name__ == "__main__":
    main()
