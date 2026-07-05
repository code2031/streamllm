"""verify_against_full_load (prompt §15.6) — streamed logits == full-load logits.

The primary guard against silent wrongness in the streaming path. Two models with
identical weights (same seed): one Tier 0 (full), one Tier 2 (streamed with a tiny
LRU so eviction is exercised). Prefill logits and multi-step greedy decode must
match.
"""

from __future__ import annotations

import torch
from tests.conftest import tiny_llama_config, tiny_qwen_config

from streamllm.model import StreamModel


def _twin_models(config_fn, seed=0):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = config_fn()
    a = AutoModelForCausalLM.from_config(cfg).eval()
    torch.manual_seed(seed)
    b = AutoModelForCausalLM.from_config(cfg).eval()
    return a, b, cfg


def _check(config_fn):
    full_model, stream_model, cfg = _twin_models(config_fn)
    full = StreamModel.from_model(full_model, cfg, tier="full", device="cpu")
    streamed = StreamModel.from_model(stream_model, cfg, tier="ram", device="cpu", cache_layers=2)
    assert streamed.runner is not None
    assert streamed.decision.tier == 2

    ids = torch.randint(0, cfg.vocab_size, (1, 7))

    # Prefill: full forward logits must match.
    lf = full.forward(ids)
    ls = streamed.forward(ids)
    assert torch.allclose(lf, ls, atol=1e-4), (lf - ls).abs().max()

    # Multi-step decode (KV cache path) greedy tokens must match.
    gf = full.generate(ids, max_new_tokens=12, do_sample=False)
    gs = streamed.generate(ids, max_new_tokens=12, do_sample=False)
    assert torch.equal(gf, gs)


def test_verify_llama():
    _check(tiny_llama_config)


def test_verify_qwen():
    _check(tiny_qwen_config)


def test_verify_holds_with_capacity_one():
    # cache_layers=1 (no double-buffer) must still be correct, just slower.
    full_model, stream_model, cfg = _twin_models(tiny_llama_config)
    full = StreamModel.from_model(full_model, cfg, tier="full", device="cpu")
    streamed = StreamModel.from_model(stream_model, cfg, tier="ram", device="cpu", cache_layers=1)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    assert torch.equal(
        full.generate(ids, max_new_tokens=8), streamed.generate(ids, max_new_tokens=8)
    )
