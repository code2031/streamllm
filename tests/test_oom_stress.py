"""Long-context OOM stress test (prompt §15.3) — the headroom guard.

A budget that *fits at load* but where KV growth over the full generation would
exceed it must NOT be chosen as Tier 0. The policy includes ``kv_bytes`` at
``max_context`` in ``tier0_peak``, so it picks a lower tier up front. The runtime
graceful-demotion half lives in ``test_demotion.py`` (needs the runner).
"""

from __future__ import annotations

from tests.conftest import make_hw, tiny_llama_config

from streamllm.config import StreamConfig
from streamllm.memory import estimate_memory
from streamllm.tiering import select_tier

# ~4.75 GB of weights at fp16 — fits an 8 GB GPU on its own (short context).
MID = dict(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=11008,
    num_hidden_layers=12,
    vocab_size=32000,
)


def _est(max_context, prompt_len=64):
    return estimate_memory(
        tiny_llama_config(**MID),
        dtype="float16",
        max_context=max_context,
        prompt_len=prompt_len,
        batch_size=1,
    )


def test_short_context_picks_tier0():
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64)
    d = select_tier(hw, _est(max_context=512), StreamConfig(), device="auto")
    assert d.tier == 0  # weights + tiny KV fit comfortably


def test_long_context_demotes_up_front_not_tier0():
    # Same weights + GPU, but KV at 100k tokens would blow the 8 GB budget.
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64)
    est = _est(max_context=50_000)
    d = select_tier(hw, est, StreamConfig(), device="auto")
    assert d.tier != 0, (
        f"KV at max_context={est.context} is {est.kv_bytes / 1e9:.1f}GB; "
        "policy must not pick Tier 0 (would OOM as KV grows)"
    )
    # And it must remain feasible (weights fit RAM) -> Tier 1, not a crash.
    assert d.tier == 1


def test_headroom_makes_borderline_fit_demote():
    # Weights+KV land just under raw free VRAM but over the 0.9 headroom line.
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64)
    est = _est(max_context=512)
    tight = StreamConfig(headroom=0.2)  # pretend almost no usable VRAM
    d = select_tier(hw, est, tight, device="auto")
    assert d.tier != 0
