"""Graceful tier demotion on real OOM (prompt §10, §15.3 runtime half).

We inject a fake CUDA OOM at materialization for the first tier and assert the
builder demotes one tier and succeeds, with a loud warning — never crashing.
"""

from __future__ import annotations

import pytest
import torch
from tests.conftest import make_hw, tiny_llama_config

import streamllm.model as model_mod
from streamllm.errors import OutOfMemoryDemotionError

SEVEN_B = dict(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=11008,
    num_hidden_layers=80,
    vocab_size=32000,
)


def test_demotion_picks_lower_tier(monkeypatch):
    # Drive _attach_with_demotion directly: Tier 1 OOMs -> demote to Tier 2.
    from streamllm.config import StreamConfig
    from streamllm.memory import estimate_memory
    from streamllm.tiering import select_tier

    cfg = tiny_llama_config(**SEVEN_B)
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64, unified=False)
    est = estimate_memory(cfg, dtype="float16", max_context=256, prompt_len=32)
    scfg = StreamConfig()
    decision = select_tier(hw, est, scfg, device="cuda")
    assert decision.tier == 1

    seq = []

    def flaky_attach(m, g, d, c, dev, e, hardware):
        seq.append(d.tier)
        if d.tier == 1:
            raise torch.cuda.OutOfMemoryError("injected OOM")
        return None

    monkeypatch.setattr(model_mod, "_attach_runtime", flaky_attach)
    final, _runner = model_mod._attach_with_demotion(
        object(),
        object(),
        est,
        scfg,
        hw,
        "cuda:0",
        "cuda",
        decision,
        cache_layers="auto",
        allow_demotion=True,
    )
    assert seq[0] == 1 and final.tier == 2  # demoted 1 -> 2
    assert final.forced  # demoted decisions are marked forced


def test_demotion_exhausted_raises(monkeypatch):
    from streamllm.config import StreamConfig
    from streamllm.memory import estimate_memory
    from streamllm.tiering import select_tier

    cfg = tiny_llama_config(**SEVEN_B)
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64, unified=False)
    est = estimate_memory(cfg, dtype="float16", max_context=256, prompt_len=32)
    scfg = StreamConfig()
    decision = select_tier(hw, est, scfg, device="cuda")

    def always_oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("injected OOM")

    monkeypatch.setattr(model_mod, "_attach_runtime", always_oom)
    with pytest.raises(OutOfMemoryDemotionError):
        model_mod._attach_with_demotion(
            object(),
            object(),
            est,
            scfg,
            hw,
            "cuda:0",
            "cuda",
            decision,
            cache_layers="auto",
            allow_demotion=True,
        )


def test_non_oom_error_propagates(monkeypatch):
    from streamllm.config import StreamConfig
    from streamllm.memory import estimate_memory
    from streamllm.tiering import select_tier

    cfg = tiny_llama_config(**SEVEN_B)
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64, unified=False)
    est = estimate_memory(cfg, dtype="float16", max_context=256, prompt_len=32)
    scfg = StreamConfig()
    decision = select_tier(hw, est, scfg, device="cuda")

    def boom(*a, **k):
        raise ValueError("not an OOM")

    monkeypatch.setattr(model_mod, "_attach_runtime", boom)
    with pytest.raises(ValueError, match="not an OOM"):
        model_mod._attach_with_demotion(
            object(),
            object(),
            est,
            scfg,
            hw,
            "cuda:0",
            "cuda",
            decision,
            cache_layers="auto",
            allow_demotion=True,
        )
