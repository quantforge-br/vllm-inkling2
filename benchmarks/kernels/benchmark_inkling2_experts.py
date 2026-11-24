# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare local Humming projections and complete Marlin expert layers on SM86."""
# ruff: noqa: B023

import argparse
import copy
import json
import os
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm import envs
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_scale_perms,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
)
from vllm.models.inkling2.nvidia.moe_ct import _pack_rows, _unpack_rows
from vllm.models.inkling2.nvidia.ops.marlin_linear import pack_marlin_w4
from vllm.models.inkling2.nvidia.ops.moe import combine
from vllm.models.inkling2.nvidia.ops.moe_sm90 import HummingExperts, align_rows, swiglu
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton
from vllm.utils.humming import CompressedTensorsWeightSchema


def log(**row):
    print(json.dumps(row), flush=True)


def checkpoint_layer(directory, rank):
    checkpoint = json.loads((directory / "config.json").read_text())
    cfg = checkpoint["text_config"]
    group = checkpoint["quantization_config"]["config_groups"]["group_0"]
    weight = group["weights"]
    assert (
        weight["num_bits"] == 4
        and not weight["symmetric"]
        and weight["group_size"] == 128
    )
    schema = CompressedTensorsWeightSchema(
        format=group["format"],
        type="int",
        num_bits=4,
        strategy="group",
        symmetric=False,
        group_size=128,
    )
    tensors = {name: {} for name in ("w13", "w2")}
    with safe_open(directory / "model.safetensors", framework="pt") as f:
        for key in ("weight_packed", "weight_scale", "weight_zero_point"):
            w13, w2 = [], []
            for expert in range(64 * rank, 64 * (rank + 1)):
                prefix = f"model.llm.layers.0.mlp.experts.{expert}."
                gate, up, down = [
                    f.get_tensor(f"{prefix}{proj}_proj.{key}").cuda()
                    for proj in ("gate", "up", "down")
                ]
                if key == "weight_zero_point":
                    gate, up = _unpack_rows(gate, 4), _unpack_rows(up, 4)
                interleaved = torch.stack((gate, up), 1).flatten(0, 1)
                w13.append(
                    _pack_rows(interleaved, 4)
                    if key == "weight_zero_point"
                    else interleaved
                )
                w2.append(down)
            for name, rows in (("w13", w13), ("w2", w2)):
                tensors[name][key] = torch.stack(rows)
    return cfg, {name: (schema, values) for name, values in tensors.items()}


def marlin_projection(tensors):
    weights, scales, zeros = [], [], []
    inv_w = get_weight_perm(4).argsort().cuda()
    inv_s = torch.tensor(get_scale_perms()[0], device="cuda").argsort()
    inv_z = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device="cuda").argsort()
    for word, s, z_word in zip(
        tensors["weight_packed"],
        tensors["weight_scale"],
        tensors["weight_zero_point"],
        strict=True,
    ):
        codes = _unpack_rows(word.T, 4).T.contiguous()
        z = _unpack_rows(z_word, 4)
        packed = pack_marlin_w4(codes, s, z)
        n, k = codes.shape
        # Decode the private layout independently of the production repacker.
        blocks = _unpack_rows(packed.T, 4).T.reshape(-1, len(inv_w))[:, inv_w]
        recovered = (
            blocks.reshape(k // 16, n // 16, 16, 16).transpose(1, 2).reshape(k, n).T
        )
        rs = packed.marlin_scale.reshape(-1, 64)[:, inv_s].reshape(k // 128, n).T
        rz = _unpack_rows(packed.marlin_zero.T, 4).T.reshape(-1, 8)[:, inv_z]
        rz = rz.reshape(-1, 64)[:, inv_s].reshape(k // 128, n).T
        assert (
            torch.equal(codes, recovered) and torch.equal(z, rz) and torch.equal(s, rs)
        )
        assert torch.equal(
            (codes.reshape(n, -1, 128) - z[:, :, None]) * s.float()[:, :, None],
            (recovered.reshape(n, -1, 128) - rz[:, :, None]) * rs.float()[:, :, None],
        )
        weights.append(packed)
        scales.append(packed.marlin_scale)
        zeros.append(packed.marlin_zero)
    return torch.stack(weights), torch.stack(scales), torch.stack(zeros)


def inputs(t, rank):
    # Both ranks receive identical input/route ids before computing local slots.
    torch.manual_seed(820 + t)
    x = torch.randn(t, 2304, device="cuda").bfloat16()
    ids = torch.randn(t, 128, device="cuda").topk(8, dim=-1).indices.int() - 64 * rank
    ids = torch.where((ids >= 0) & (ids < 64), ids, -1)
    weights = torch.randn(t, 8, device="cuda").softmax(1).flatten()
    src = torch.where(
        ids.flatten() >= 0, torch.arange(8 * t, device="cuda", dtype=torch.int32), -1
    )
    return x, ids, weights, src


def measure(fn, rep=8):
    return 1000 * triton.testing.do_bench_cudagraph(fn, rep=rep)


def marlin_layer(cfg, engine, packed, rank, t):
    x, ids, gamma, src = inputs(t, rank)
    workspace = marlin_make_workspace_new(x.device, 4)

    def gemm(key, a, alignment, topk):
        w, s, z = packed[key]
        n, k = engine.configs[key].shape_n, a.shape[1]
        return ops.moe_wna16_marlin_gemm(
            a,
            None,
            w,
            None,
            s,
            None,
            z,
            workspace,
            *alignment,
            gamma,
            8,
            topk,
            False,
            scalar_types.uint4,
            len(a),
            n,
            k,
            False,
            True,
            False,
        )

    def candidate():
        alignment = align_rows(ids, 8)
        gateup = gemm("w13", x, alignment, 8)
        h = swiglu(gateup, gamma, src, cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"])
        return combine(gemm("w2", h, alignment, 1), src, 8, x.dtype)

    def baseline():
        alignment = align_rows(ids, engine._block_m("w13", 4 * t)) if t == 1 else None
        down = engine(
            x,
            ids,
            gamma,
            src,
            128,
            cfg["mlp_gate_softcap"],
            cfg["mlp_up_softcap"],
            alignment,
        )
        return combine(down, src, 8, x.dtype)

    a, b = baseline(), candidate()
    diff = (a.float() - b.float()).abs()
    assert torch.isfinite(diff).all()
    memory = {}
    for name, fn in (("humming", baseline), ("marlin", candidate)):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        memory[name] = torch.cuda.max_memory_allocated() - allocated
    log(
        experiment="layer",
        rank=rank,
        t=t,
        humming_us=measure(baseline),
        marlin_us=measure(candidate),
        temporary_bytes=memory,
        max_error=diff.max().item(),
        rel_l2=(diff.norm() / a.float().norm()).item(),
        unequal=diff.count_nonzero().item(),
    )


def humming_tiles(engine, rank, t, stream_k):
    x, ids, gamma, src = inputs(t, rank)
    for name, data, topk in (
        ("w13", x, 8),
        ("w2", torch.randn(t * 8, 1152, device="cuda").bfloat16(), 1),
    ):
        tuning = engine.tunings[name]
        lower, upper, chosen = next(row for row in tuning if row[0] < 4 * t <= row[1])
        original = copy.deepcopy(chosen)

        def configure(update):
            chosen.clear()
            chosen.update(original | update)
            engine.tuning_strs[name] = json.dumps(tuning)
            return align_rows(ids, chosen["block_shape"][0])

        def run():
            return engine._gemm(name, data, t * 8, alignment, topk, 4 * t)

        alignment = configure({})
        local = src >= 0
        reference = run()[local]
        configs = [{}]
        if stream_k:
            assert not envs.VLLM_BATCH_INVARIANT
            configs += [dict(use_stream_k=True)]
        else:
            for bm, bn, bk in ((8, 64, 128), (8, 128, 128), (8, 64, 64), (16, 64, 128)):
                for warp_n, warp_k, stages, ctas in (
                    (64, 64, 2, 3),
                    (32, 64, 2, 3),
                    (64, 32, 3, 2),
                    (32, 32, 3, 4),
                ):
                    configs.append(
                        dict(
                            block_shape=[bm, bn, bk],
                            warp_shape=[bm, warp_n, warp_k],
                            num_stages=stages,
                            num_ctas_per_sm=ctas,
                        )
                    )
        try:
            for update in configs:
                alignment = configure(update)
                row = dict(
                    experiment="tiles",
                    rank=rank,
                    t=t,
                    projection=name,
                    bucket=[lower, upper],
                    valid_rows=4 * t,
                    local=int(local.sum()),
                    padded=int(alignment[2]),
                    config=chosen.copy(),
                    baseline=not update,
                )
                try:
                    out = run()[local]
                    times = [measure(run, 3) for _ in range(3)]
                    log(
                        **row,
                        us=statistics.median(times),
                        repeats_us=times,
                        exact=torch.equal(reference, out),
                        max_error=(out.float() - reference.float()).abs().max().item(),
                        max_repeat_error=max(
                            (run()[local].float() - out.float()).abs().max().item()
                            for _ in range(5)
                        ),
                    )
                except (RuntimeError, ValueError, AssertionError) as exc:
                    log(**row, error=str(exc))
        finally:
            configure({})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--experiment", choices=("tiles", "stream-k", "layer"), required=True
    )
    parser.add_argument("--rank", type=int, choices=(0, 1), default=None)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"]) if args.rank is None else args.rank
    if args.rank is None:
        torch.cuda.set_device(rank)
    cfg, weights = checkpoint_layer(args.checkpoint, rank)
    shapes = ((2304, 2304), (2304, 1152))
    packed = {}
    if args.experiment == "layer":
        packed = {
            key: marlin_projection(fields) for key, (_, fields) in weights.items()
        }
        log(
            experiment="repack",
            rank=rank,
            experts=64,
            projections=2,
            dequant_exact=True,
        )
    engine = HummingExperts(weights, shapes, False)
    for t in range(1, 9):
        if args.experiment == "layer":
            marlin_layer(cfg, engine, packed, rank, t)
        else:
            humming_tiles(engine, rank, t, args.experiment == "stream-k")


if __name__ == "__main__":
    main()
