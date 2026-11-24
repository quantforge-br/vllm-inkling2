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
