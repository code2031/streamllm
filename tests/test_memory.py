"""Estimator property tests (prompt §15.2)."""

from __future__ import annotations

import pytest
from tests.conftest import tiny_llama_config

from streamllm.memory import count_params_from_config, estimate_memory


def _est(cfg, **kw):
    base = dict(dtype="float16", max_context=128, prompt_len=16, batch_size=1)
    base.update(kw)
    return estimate_memory(cfg, **base)


def test_kv_scales_linearly_in_context():
    cfg = tiny_llama_config()
    e1 = _est(cfg, max_context=128)
    e2 = _est(cfg, max_context=256)
    assert e2.kv_bytes == pytest.approx(2 * e1.kv_bytes, rel=1e-9)


def test_kv_scales_linearly_in_batch():
    cfg = tiny_llama_config()
    e1 = _est(cfg, batch_size=1)
    e4 = _est(cfg, batch_size=4)
    assert e4.kv_bytes == pytest.approx(4 * e1.kv_bytes, rel=1e-9)


def test_kv_scales_linearly_in_layers():
    e_small = _est(tiny_llama_config(num_hidden_layers=2))
    e_big = _est(tiny_llama_config(num_hidden_layers=8))
    assert e_big.kv_bytes == pytest.approx(4 * e_small.kv_bytes, rel=1e-9)


def test_gqa_yields_proportionally_smaller_kv():
    # 4 attn heads, 4 KV heads (no GQA) vs 4 attn heads, 1 KV head (4x smaller).
    full = _est(tiny_llama_config(num_attention_heads=4, num_key_value_heads=4))
    gqa = _est(tiny_llama_config(num_attention_heads=4, num_key_value_heads=1))
    assert full.kv_bytes == pytest.approx(4 * gqa.kv_bytes, rel=1e-9)


def test_tied_embeddings_not_double_counted():
    untied = count_params_from_config(tiny_llama_config(tie_word_embeddings=False))
    tied = count_params_from_config(tiny_llama_config(tie_word_embeddings=True))
    # Untied has an extra vocab*hidden lm_head; tied shares the embedding.
    delta = untied.resident_params - tied.resident_params
    assert delta == untied.vocab_size * untied.hidden_size


def test_sliding_window_caps_kv():
    capped = _est(tiny_llama_config(sliding_window=32), max_context=128)
    uncapped = _est(tiny_llama_config(), max_context=128)
    # KV length is min(context, window) = 32 vs 128 -> 4x smaller.
    assert uncapped.kv_bytes == pytest.approx(4 * capped.kv_bytes, rel=1e-9)


def test_quant_shrinks_weight_bytes_only():
    fp16 = _est(tiny_llama_config(), dtype="float16", quantize=None)
    int4 = _est(tiny_llama_config(), dtype="float16", quantize="int4")
    # int4 = 0.5 B/param vs fp16 2 B/param -> 4x smaller weights.
    assert fp16.per_layer_bytes == pytest.approx(4 * int4.per_layer_bytes, rel=1e-9)
    # KV is unaffected by weight quant (activations not quantized).
    assert fp16.kv_bytes == int4.kv_bytes


def test_measured_override_marks_source():
    cfg = tiny_llama_config()
    e = estimate_memory(
        cfg,
        dtype="float16",
        max_context=128,
        prompt_len=16,
        per_layer_params=12345,
        resident_params=678,
    )
    assert e.source == "measured"
    assert e.per_layer_params == 12345


def test_moe_detected_and_counted():
    cfg = tiny_llama_config()
    cfg.num_local_experts = 8
    cfg.moe_intermediate_size = 64
    counts = count_params_from_config(cfg)
    assert counts.is_moe
    assert counts.num_experts == 8


def test_activation_scales_with_prompt_and_batch():
    e1 = _est(tiny_llama_config(), prompt_len=16, batch_size=1)
    e2 = _est(tiny_llama_config(), prompt_len=32, batch_size=2)
    assert e2.activation_bytes == pytest.approx(4 * e1.activation_bytes, rel=1e-9)
