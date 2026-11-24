# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading of the Inkling2 DSpark pruned draft vocabulary."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.inkling2.nvidia.dspark import DRAFT_VOCAB_FILE, _load_draft_vocab


def _configs(model_dir, pruned_vocab_size=None):
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(model=str(model_dir), revision=None)
        )
    )
    return vllm_config, SimpleNamespace(pruned_vocab_size=pruned_vocab_size)


def test_full_vocab_without_pruned_vocab_size(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_INKLING2_DRAFT_VOCAB_PATH", raising=False)
    torch.save(torch.arange(4, dtype=torch.int32), tmp_path / DRAFT_VOCAB_FILE)
    assert _load_draft_vocab(*_configs(tmp_path)) is None


def test_checkpoint_vocab_is_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_INKLING2_DRAFT_VOCAB_PATH", raising=False)
    ids = torch.tensor([0, 3, 7, 9], dtype=torch.int32)
    torch.save(ids, tmp_path / DRAFT_VOCAB_FILE)
    assert torch.equal(_load_draft_vocab(*_configs(tmp_path, 4)), ids)


def test_checkpoint_vocab_size_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_INKLING2_DRAFT_VOCAB_PATH", raising=False)
    torch.save(torch.arange(4, dtype=torch.int32), tmp_path / DRAFT_VOCAB_FILE)
    with pytest.raises(ValueError, match="pruned_vocab_size=6"):
        _load_draft_vocab(*_configs(tmp_path, 6))


def test_checkpoint_vocab_missing_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_INKLING2_DRAFT_VOCAB_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match=DRAFT_VOCAB_FILE):
        _load_draft_vocab(*_configs(tmp_path, 4))


def test_env_path_overrides_checkpoint(tmp_path, monkeypatch):
    override = tmp_path / "override"
    override.mkdir()
    ids = torch.tensor([1, 2], dtype=torch.int64)
    torch.save(ids, override / DRAFT_VOCAB_FILE)
    torch.save(torch.arange(4, dtype=torch.int32), tmp_path / DRAFT_VOCAB_FILE)
    monkeypatch.setenv("VLLM_INKLING2_DRAFT_VOCAB_PATH", str(override))
    assert torch.equal(_load_draft_vocab(*_configs(tmp_path, 4)), ids)
