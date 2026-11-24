# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPTQ Hessian collection and the symmetric W4 quantizer of Inkling2."""

import pytest
import torch

import vllm.models.inkling2.nvidia.ops.gptq as gptq


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_INKLING2_GPTQ_COLLECT", str(tmp_path))
    monkeypatch.setenv("VLLM_INKLING2_GPTQ_COLLECT_KINDS", "experts")
    monkeypatch.setenv("VLLM_INKLING2_GPTQ_SAVE_EVERY", "0")
    for fn in (gptq.collect_dir, gptq.collect_kinds, gptq.save_every):
        fn.cache_clear()
    (tmp_path / "ACTIVE").touch()
    yield gptq._Collector()
    for fn in (gptq.collect_dir, gptq.collect_kinds, gptq.save_every):
        fn.cache_clear()


@pytest.mark.parametrize("per_slot", [False, True])
def test_add_experts_matches_naive_per_expert(collector, per_slot):
    torch.manual_seed(0)
    tokens, topk, experts, k = 37, 4, 6, 16
    ids = torch.randint(-1, experts, (tokens, topk))
    rows = tokens * topk if per_slot else tokens
    x = torch.randn(rows, k)
    scale = torch.rand(tokens * topk)
    for _ in range(2):  # accumulates over calls (loop passes)
        collector.add_experts("e", x, ids, scale, experts, per_slot)
    want = torch.zeros(experts, k, k)
    count = torch.zeros(experts, dtype=torch.int64)
    for t in range(tokens):
        for j in range(topk):
            e = int(ids[t, j])
            if e < 0:
                continue
            slot = t * topk + j
            row = (x[slot] if per_slot else x[t]) * scale[slot]
            want[e] += 2 * torch.outer(row, row)
            count[e] += 2
    torch.testing.assert_close(collector.hessians["e"], want, rtol=1e-5, atol=1e-5)
    assert torch.equal(collector.counts["e"], count)


def test_collector_skips_until_active(collector, tmp_path):
    (tmp_path / "ACTIVE").unlink()
    ids = torch.zeros(3, 2, dtype=torch.long)
    collector.add_experts("e", torch.ones(3, 8), ids, None, 2, False)
    assert "e" not in collector.hessians


def _problem(batch, n, k, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(batch, n, k) * 0.02
    w[:, :, 5] *= 8  # an outlier column
    x = torch.randn(batch, 4 * k, k) * torch.linspace(0.2, 3, k)
    h = x.transpose(1, 2) @ x
    return w, h


def _output_error(w, codes, scales, h):
    d = gptq.dequantize_w4_sym(codes, scales) - w
    return torch.einsum("bnk,bkl,bnl->", d, h, d).item()


def test_gptq_w4_sym_codes_are_uint4b8_and_beat_round_to_nearest():
    w, h = _problem(2, 32, 256)
    codes, scales = gptq.gptq_w4_sym(w, h)
    assert codes.dtype == torch.uint8 and int(codes.max()) <= 15
    assert scales.dtype == torch.bfloat16 and bool((scales > 0).all())
    amax = w.view(2, 32, 2, 128).abs().amax(-1)
    rtn_scales = (amax / 7.5).bfloat16()
    rtn = torch.clamp(
        torch.round(w.view(2, 32, 2, 128) / rtn_scales.float().unsqueeze(-1)), -8, 7
    )
    rtn_codes = (rtn + 8).to(torch.uint8).view(2, 32, 256)
    assert _output_error(w, codes, scales, h) < 0.7 * _output_error(
        w, rtn_codes, rtn_scales, h
    )


def test_gptq_w4_sym_batch_matches_single_matrices():
    w, h = _problem(3, 16, 256, seed=1)
    codes, scales = gptq.gptq_w4_sym(w, h)
    for b in range(3):
        c1, s1 = gptq.gptq_w4_sym(w[b], h[b])
        # Batched and single GEMMs may round differently near ties.
        assert (c1 != codes[b]).float().mean() < 0.01
        torch.testing.assert_close(s1, scales[b], rtol=1e-2, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pack_marlin_w4_sym_matches_dequantized_matmul():
    from vllm.models.inkling2.nvidia.ops.marlin_linear import (
        marlin_linear,
        marlin_parameter,
    )

    torch.manual_seed(0)
    n, k = 200, 512  # n pads to 256
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device="cuda")
    scales = (torch.rand(n, k // 128, device="cuda") * 0.01 + 1e-3).bfloat16()
    weight = marlin_parameter(None, (codes, scales))
    x = torch.randn(5, k, device="cuda", dtype=torch.bfloat16)
    want = x.float() @ gptq.dequantize_w4_sym(codes, scales).t()
    got = marlin_linear(x, weight).float()
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_unpack_ct_int4_inverts_compressed_tensors_packing():
    from vllm.models.inkling2.nvidia.moe_ct import _pack_cols
    from vllm.models.inkling2.nvidia.ops.marlin_linear import unpack_ct_int4

    torch.manual_seed(0)
    q = torch.randint(-8, 8, (24, 256), dtype=torch.int8)
    codes = unpack_ct_int4(_pack_cols(q, 4))
    assert codes.dtype == torch.uint8
    assert torch.equal(codes.to(torch.int16) - 8, q.to(torch.int16))


def _ct_tensors(n: int, k: int, device: str):
    """Random symmetric int4 codes, positive scales and their CT tensors."""
    from vllm.models.inkling2.nvidia.moe_ct import _pack_cols

    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device=device)
    scales = (torch.rand(n, k // 128, device=device) * 0.01 + 1e-3).bfloat16()
    packed = _pack_cols(codes.to(torch.int8) - 8, 4)
    return codes, scales, packed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_packed_int4_layers_load_shards_into_marlin(dist_init, default_vllm_config):
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.models.inkling2.nvidia.ops.linear import linear
    from vllm.models.inkling2.nvidia.ops.marlin_linear import (
        finalize_packed_int4,
        register_packed_int4,
    )

    torch.manual_seed(0)
    device, k = "cuda", 256
    with torch.device(device):
        merged = MergedColumnParallelLinear(k, [192, 24, 8], bias=False)
        qkv = QKVParallelLinear(k, 64, 2, 1, bias=False)
        row = RowParallelLinear(k, 128, bias=False, reduce_results=False)
    cases = []
    for layer, shards in (
        (merged, [(0, 192), (1, 24)]),  # the last output (padding) stays empty
        (qkv, [("q", 128), ("k", 64), ("v", 64)]),
        (row, [(None, 128)]),
    ):
        register_packed_int4(layer)
        blocks = []
        for shard, n in shards:
            codes, scales, packed = _ct_tensors(n, k, device)
            for name, tensor in (("weight_packed", packed), ("weight_scale", scales)):
                param = getattr(layer, name)
                if shard is None:
                    param.weight_loader(param, tensor)
                else:
                    param.weight_loader(param, tensor, shard)
            blocks.append(gptq.dequantize_w4_sym(codes, scales))
        finalize_packed_int4(layer)
        cases.append((layer, torch.cat(blocks)))
    x = torch.randn(3, k, device=device, dtype=torch.bfloat16)
    for layer, want in cases:
        got = linear(x, layer.weight).float()
        assert not hasattr(layer, "weight_packed")
        torch.testing.assert_close(
            got[:, : want.shape[0]], x.float() @ want.t(), rtol=2e-2, atol=2e-2
        )
        if got.shape[1] > want.shape[0]:
            assert not got[:, want.shape[0] :].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_int8_embedding_matches_dequantized_table(dist_init, default_vllm_config):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.inkling2.nvidia.ops.embedding_int8 import (
        make_embedding_int8,
        quantize_embedding_int8,
    )

    torch.manual_seed(0)
    with torch.device("cuda"):
        embed = VocabParallelEmbedding(1000, 256)
    embed.weight.data.normal_(0, 0.02)
    table = embed.weight.data.clone()
    codes, scales = quantize_embedding_int8(table)
    rows = table.shape[0]  # padded to a multiple of 64
    want_table = (codes.float().view(rows, 2, 128) * scales.float().unsqueeze(-1)).view(
        rows, 256
    )
    assert (want_table - table.float()).abs().max() <= table.abs().max() / 127
    make_embedding_int8(embed)
    ids = torch.tensor([[0, 999, 5], [7, 7, 123]], device="cuda")
    got = embed(ids)
    assert got.shape == (2, 3, 256) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want_table[ids].bfloat16().float())
