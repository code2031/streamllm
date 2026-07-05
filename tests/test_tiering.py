"""Tier selection tests (prompt §15.1) — each tier reachable + deciding math."""

from __future__ import annotations

import pytest
from tests.conftest import make_hw, tiny_llama_config

from streamllm.config import StreamConfig
from streamllm.memory import estimate_memory
from streamllm.tiering import demotion_ladder, normalize_tier, select_tier

GiB = 1024**3


def _est(**kw):
    cfg_over = kw.pop("cfg_over", {})
    base = dict(dtype="float16", max_context=256, prompt_len=32, batch_size=1)
    base.update(kw)
    return estimate_memory(tiny_llama_config(**cfg_over), **base)


# Realistically-sized configs (estimate_memory is pure arithmetic, no allocation),
# so each lands unambiguously in a target tier under the mocked budgets below.
SEVEN_B = dict(  # ~7B: 28 GB weights at fp16
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=11008,
    num_hidden_layers=80,
    vocab_size=32000,
)
SEVENTY_B_LAYER = dict(  # ~1.7 GB per layer -> a 4 GB GPU can't hold 2 layers
    hidden_size=8192,
    num_attention_heads=64,
    num_key_value_heads=8,
    intermediate_size=28672,
    num_hidden_layers=80,
    vocab_size=32000,
)
HUGE = dict(  # weights far exceed any RAM budget -> disk
    hidden_size=16384,
    num_attention_heads=128,
    num_key_value_heads=8,
    intermediate_size=53248,
    num_hidden_layers=126,
    vocab_size=128000,
)


def test_tier0_when_fits_vram():
    est = _est()
    hw = make_hw(cuda=True, vram_total_gb=24, vram_free_gb=24, ram_avail_gb=64)
    d = select_tier(hw, est, StreamConfig(), device="auto")
    assert d.tier == 0
    assert "vLLM" in (d.honest_note or "")


def test_tier1_gpu_ram_offload():
    # ~7B (28 GB) doesn't fit an 8 GB discrete GPU but fits 64 GB RAM, with room
    # for >=2 resident layers on the GPU -> Tier 1.
    est = _est(cfg_over=SEVEN_B)
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64, unified=False)
    d = select_tier(hw, est, StreamConfig(), device="auto")
    assert d.tier == 1
    assert d.cache_layers is not None and d.cache_layers >= 2
    assert d.backing == "ram"


def test_tier2_small_gpu_cannot_hold_two_layers():
    # ~1.7 GB/layer on a 4 GB GPU: can't keep 2 resident layers (cache_dev<2),
    # so Tier 1 is infeasible; weights fit big RAM -> Tier 2 streaming into GPU.
    est = _est(cfg_over=SEVENTY_B_LAYER)
    hw = make_hw(cuda=True, vram_total_gb=4, vram_free_gb=4, ram_avail_gb=400, unified=False)
    d = select_tier(hw, est, StreamConfig(), device="auto")
    assert d.tier == 2
    assert d.compute_device == "cuda:0"


def test_tier2_unified_memory():
    # Unified memory (GB10-style) that can't hold the whole model at Tier-0 peak
    # in its small usable budget, but the weights fit in RAM -> Tier 2.
    est = _est(cfg_over=SEVEN_B)
    hw = make_hw(
        cuda=True,
        vram_total_gb=128,
        vram_free_gb=4,
        ram_total_gb=128,
        ram_avail_gb=96,
        unified=True,
    )
    d = select_tier(hw, est, StreamConfig(), device="auto")
    assert d.tier == 2
    assert "Unified memory" in (d.honest_note or "")


def test_tier3_disk_when_exceeds_ram():
    est = _est(cfg_over=HUGE)
    hw = make_hw(cuda=False, ram_total_gb=64, ram_avail_gb=32, disk_free_gb=2000)
    d = select_tier(hw, est, StreamConfig(), device="auto")
    assert d.tier == 3
    assert d.backing == "disk"
    assert "I/O-bound" in (d.honest_note or "")


def test_tier_override_forces_tier():
    est = _est()
    hw = make_hw(cuda=True, vram_total_gb=24, vram_free_gb=24, ram_avail_gb=64)
    d = select_tier(hw, est, StreamConfig(), device="auto", tier_override="disk")
    assert d.tier == 3
    assert d.forced


def test_normalize_tier_aliases():
    assert normalize_tier("auto") is None
    assert normalize_tier("full") == 0
    assert normalize_tier("gpu_ram") == 1
    assert normalize_tier("ram") == 2
    assert normalize_tier("disk") == 3
    assert normalize_tier(2) == 2


def test_normalize_tier_rejects_bad():
    with pytest.raises(Exception):
        normalize_tier(9)
    with pytest.raises(Exception):
        normalize_tier("nonsense")


def test_demotion_ladder():
    assert demotion_ladder(0) == [1, 2, 3]
    assert demotion_ladder(2) == [3]
    assert demotion_ladder(3) == []


def test_cache_layers_math_matches_formula():
    cfg = StreamConfig()
    est = _est(cfg_over=SEVEN_B)
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64, unified=False)
    d = select_tier(hw, est, cfg, device="auto")
    assert d.tier == 1
    overhead = cfg.cuda_overhead_reserve_bytes
    usable = int(8 * GiB * cfg.headroom)
    expected = (
        usable - est.resident_bytes - est.kv_bytes - est.activation_bytes - overhead
    ) // est.per_layer_bytes
    assert d.numbers["cache_layers"] == expected
