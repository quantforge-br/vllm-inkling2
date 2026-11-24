# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare reference and M=1/K=256 Markov output projection on SM86."""

import argparse
import json

import torch

from vllm.models.inkling2.nvidia.ops.int8_linear import (
    int8_linear,
    markov_m1_int8,
    pack_int8_rows,
)
from vllm.triton_utils import triton


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True, help="Physical TP-local N")
    parser.add_argument("--seeds", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(1)
    packed = pack_int8_rows(torch.randn(args.rows, 256, device=device).bfloat16())
    records = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        x = torch.randn(1, 256, device=device).bfloat16()
        expected, actual = int8_linear(x, packed), markov_m1_int8(x, packed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = markov_m1_int8(x, packed)
        graph.replay()
        torch.testing.assert_close(actual, replay, atol=0, rtol=0)
        # Scale with the reference temperature that places each winning
        # probability immediately below or above the deployed threshold.
        lo, hi = 1e-3, 8.0
        ref, lab = expected.float(), actual.float()
        for _ in range(20):
            temperature = (lo + hi) / 2
            if torch.softmax(ref / temperature, dim=-1).amax().item() > 0.5:
                lo = temperature
            else:
                hi = temperature
        record = {
            "seed": seed,
            "expected_id": expected.argmax().item(),
            "actual_id": actual.argmax().item(),
            "mismatches": (expected != actual).count_nonzero().item(),
            "max_abs": (ref - lab).abs().amax().item(),
            "confidence": [
                [
                    temp,
                    torch.softmax(ref / temp, dim=-1).amax().item(),
                    torch.softmax(lab / temp, dim=-1).amax().item(),
                ]
                for temp in (lo * 0.999, hi * 1.001)
            ],
        }
        records.append(record)
    print(
        json.dumps(
            {
                "rows": args.rows,
                "reference_us": triton.testing.do_bench_cudagraph(
                    lambda: int8_linear(x, packed), rep=100
                )
                * 1000,
                "candidate_us": triton.testing.do_bench_cudagraph(
                    lambda: markov_m1_int8(x, packed), rep=100
                )
                * 1000,
                "cases": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
