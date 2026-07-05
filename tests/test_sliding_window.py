"""Sliding-window attention: estimator cap + real streamed-vs-full forward.

Mistral caps its KV cache at the sliding window. The estimator must reflect that,
and the streamed forward must still match a full load when the model's own
sliding-window mask runs (the runner never touches the mask).
"""

from __future__ import annotations

import torch


def _mistral(seed=0, sliding_window=8, layers=2):
    from transformers import AutoModelForCausalLM, MistralConfig

    torch.manual_seed(seed)
    cfg = MistralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=sliding_window,
        max_position_embeddings=128,
    )
    return AutoModelForCausalLM.from_config(cfg).eval(), cfg


def test_estimator_caps_kv_at_window():
    from streamllm.memory import estimate_memory

    _, cfg = _mistral(sliding_window=8)
    capped = estimate_memory(cfg, dtype="float16", max_context=128, prompt_len=16)
    # KV length is min(128, 8) = 8; a no-window version at 128 would be 16x bigger.
    cfg.sliding_window = None
    uncapped = estimate_memory(cfg, dtype="float16", max_context=128, prompt_len=16)
    assert uncapped.kv_bytes == 16 * capped.kv_bytes
    assert capped.sliding_window == 8


def test_streamed_matches_full_with_sliding_window():
    from streamllm.model import StreamModel

    full_model, cfg = _mistral(seed=1)
    stream_model, _ = _mistral(seed=1)  # identical weights
    full = StreamModel.from_model(full_model, cfg, tier="full", device="cpu")
    streamed = StreamModel.from_model(stream_model, cfg, tier="ram", device="cpu", cache_layers=1)

    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    # Generate past the 8-token window so the sliding-window mask actually engages.
    gf = full.generate(ids, max_new_tokens=10, do_sample=False)
    gs = streamed.generate(ids, max_new_tokens=10, do_sample=False)
    assert torch.equal(gf, gs)
