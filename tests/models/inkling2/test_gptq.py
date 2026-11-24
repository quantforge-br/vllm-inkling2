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
