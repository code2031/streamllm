"""Shard round-trip + resumability (prompt §15.9)."""

from __future__ import annotations

import torch
from tests.conftest import tiny_llama_config

from streamllm.model import StreamModel
from streamllm.shard import load_manifest, shard_model, verify_shards


def _model(seed=0, **over):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = tiny_llama_config(**over)
    return AutoModelForCausalLM.from_config(cfg).eval(), cfg


def test_shard_round_trip_logits_match(tmp_path):
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        ref = model(ids).logits

    res = shard_model(model, cfg, out_path=tmp_path)
    assert (tmp_path / "streamllm_manifest.json").exists()
    assert (tmp_path / "resident.safetensors").exists()
    assert len(res.manifest["layers"]) == cfg.num_hidden_layers
    assert verify_shards(tmp_path)  # all hashes match

    sm = StreamModel.from_shards(tmp_path, device="cpu")
    assert sm.decision.tier == 3
    with torch.no_grad():
        got = sm.forward(ids)
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


def test_shard_manifest_records_hashes_and_dtype(tmp_path):
    model, cfg = _model()
    shard_model(model, cfg, out_path=tmp_path, source_model="dummy/model")
    m = load_manifest(tmp_path)
    assert m["source_model"] == "dummy/model"
    assert m["num_layers"] == cfg.num_hidden_layers
    for layer in m["layers"]:
        assert len(layer["sha256"]) == 64
        assert layer["bytes"] > 0


def test_shard_resumable_only_rebuilds_missing(tmp_path):
    model, cfg = _model()
    r1 = shard_model(model, cfg, out_path=tmp_path)
    assert len(r1.built) == cfg.num_hidden_layers
    assert len(r1.skipped) == 0

    # Corrupt-by-deletion of one shard; re-shard should rebuild ONLY that one.
    (tmp_path / "layer_0001.safetensors").unlink()
    r2 = shard_model(model, cfg, out_path=tmp_path)
    assert r2.built == ["layer_0001.safetensors"]
    assert len(r2.skipped) == cfg.num_hidden_layers - 1


def test_corrupt_shard_detected(tmp_path):
    model, cfg = _model()
    shard_model(model, cfg, out_path=tmp_path)
    # Flip bytes in a shard -> verify must fail.
    p = tmp_path / "layer_0000.safetensors"
    data = bytearray(p.read_bytes())
    data[-1] ^= 0xFF
    p.write_bytes(bytes(data))
    import pytest

    from streamllm.errors import ShardError

    with pytest.raises(ShardError, match="hash mismatch"):
        verify_shards(tmp_path)


def test_insufficient_disk_refused(tmp_path, monkeypatch):
    import shutil

    model, cfg = _model()

    class _Usage:
        total = 10**12
        free = 1  # 1 byte free

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage())
    import pytest

    from streamllm.errors import ShardError

    with pytest.raises(ShardError, match="insufficient disk"):
        shard_model(model, cfg, out_path=tmp_path)
