# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise W8 Marlin with the checkpoint's actual vocabulary shards on SM86."""
# ruff: noqa: B023

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
    marlin_weights,
)
from vllm.models.inkling2.nvidia.ops.int8_linear import (
    int8_linear,
    pack_int8_rows,
)
from vllm.models.inkling2.nvidia.ops.marlin_linear import (
    marlin_linear,
    prepare_marlin_w8,
)
from vllm.triton_utils import triton


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    valid_vocab = cfg["vocab_size"]
    torch.manual_seed(37)
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as f:
        shard = f.get_slice("model.llm.unembed.weight")
        total, k = shard.get_shape()
        n = total // 2
        for rank in (0, 1):
            w = shard[rank * n : (rank + 1) * n].cuda().bfloat16()
            packed = pack_int8_rows(w)
            prepare_marlin_w8(w, packed)
            candidate = w.marlin_w8
            scales = packed[:, k : k + 4].contiguous().view(torch.float32)
            assert torch.equal(w.marlin_row_scales, scales.bfloat16())
            # Independent layout oracle on the final tile (also covers added padding).
            padded_n = candidate.marlin_scale.shape[1]
            start = padded_n - 64
            codes = torch.full((64, k), 128, dtype=torch.int32, device="cuda")
            codes[: n - start] = packed[start:, :k].int() + 128
            expected = marlin_weights(
                codes.t().contiguous(), k, 64, 8, get_weight_perm(8)
            )
            assert torch.equal(candidate[:, -expected.shape[1] :], expected)
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        n=n,
                        padded_n=padded_n,
                        k=k,
                        changed_bf16_scales=(scales != w.marlin_row_scales)
                        .sum()
                        .item(),
                        max_scale_delta=(scales - w.marlin_row_scales)
                        .abs()
                        .max()
                        .item(),
                        max_dequantized_weight_delta=(
                            (scales - w.marlin_row_scales).abs().flatten()
                            * packed[:, :k].abs().amax(1)
                        )
                        .max()
                        .item(),
                    )
                ),
                flush=True,
            )
            for m in range(1, 9):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

                def reference():
                    return int8_linear(x, packed, out_dtype=torch.float32)

                def marlin():
                    return marlin_linear(x, candidate, out_dtype=torch.float32)

                a, b = reference(), marlin()
                valid = min(n, valid_vocab - rank * n)
                assert a.shape == b.shape == (m, n)
                for ties in (x[:1] * 0,):
                    tied = marlin_linear(ties, candidate)
                    assert (
                        tied.shape == (1, n)
                        and tied.abs().max() == 0
                        and tied.argmax() == 0
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay = marlin()
                graph.replay()
                assert torch.equal(replay, b)
                print(
                    json.dumps(
                        dict(
                            rank=rank,
                            m=m,
                            valid=valid,
                            argmax_equal=(
                                a[:, :valid].argmax(1) == b[:, :valid].argmax(1)
                            )
                            .sum()
                            .item(),
                            max_error=(a - b).abs().max().item(),
                            mean_error=(a - b).abs().mean().item(),
                            original_us=1000
                            * triton.testing.do_bench_cudagraph(reference, rep=15),
                            marlin_us=1000
                            * triton.testing.do_bench_cudagraph(marlin, rep=15),
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
