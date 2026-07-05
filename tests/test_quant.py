"""Quantization round-trip + quantized-shard size (prompt §9)."""

from __future__ import annotations

import torch
from tests.conftest import tiny_llama_config

from streamllm.model import StreamModel
from streamllm.quant import dequantize_tensor, maybe_quantize_state
from streamllm.shard import shard_model


class _FakeHandle:
    """Stands in for a safetensors handle: serves the scale tensors by key."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._t = tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._t[name]


def test_int8_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(16, 24)
    out, meta = maybe_quantize_state({"weight": w}, "int8")
    handle = _FakeHandle(out)
    deq = dequantize_tensor(out["weight"], handle, "weight", meta)
    scale = out["weight__scale"]  # per-row
    # Symmetric per-channel int8: |error| <= scale/2 elementwise.
    assert torch.all((w - deq).abs() <= scale + 1e-6)
    assert out["weight"].dtype == torch.int8


def test_int4_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(16, 24)
    out, meta = maybe_quantize_state({"weight": w}, "int4")
    handle = _FakeHandle(out)
    deq = dequantize_tensor(out["weight"], handle, "weight", meta)
    scale = out["weight__scale"]
    assert deq.shape == w.shape
    assert torch.all((w - deq).abs() <= scale + 1e-6)
    # Packed nibbles: ~half the element count of an int8 store.
    assert out["weight"].dtype == torch.uint8


def test_1d_tensors_not_quantized():
    out, meta = maybe_quantize_state({"norm": torch.randn(32)}, "int8")
    assert out["norm"].dtype == torch.float32
    assert "norm" not in meta["params"]


def test_quantized_shards_are_smaller(tmp_path):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = tiny_llama_config(hidden_size=64, intermediate_size=128, vocab_size=128)
    model = AutoModelForCausalLM.from_config(cfg).eval()

    full = shard_model(model, cfg, out_path=tmp_path / "fp")
    i8 = shard_model(model, cfg, out_path=tmp_path / "i8", quantize="int8")
    i4 = shard_model(model, cfg, out_path=tmp_path / "i4", quantize="int4")
    assert i8.total_bytes < full.total_bytes
    assert i4.total_bytes < i8.total_bytes


def test_int8_sharded_reload_runs(tmp_path):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = tiny_llama_config(hidden_size=64, intermediate_size=128, vocab_size=128)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    shard_model(model, cfg, out_path=tmp_path, quantize="int8")
    sm = StreamModel.from_shards(tmp_path, device="cpu")
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    logits = sm.forward(ids)
    assert logits.shape == (1, 5, cfg.vocab_size)
    assert torch.isfinite(logits).all()
