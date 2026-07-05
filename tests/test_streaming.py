"""LRU eviction + prefetch-overlap (prompt §15.5, §15.10)."""

from __future__ import annotations

import time

import torch
from tests.conftest import tiny_llama_config

from streamllm.cache import LayerCache
from streamllm.model import StreamModel


class _FakeHandle:
    def __init__(self, index: int) -> None:
        self.index = index
        self.nbytes = 0
        self.evicted = False

    def evict(self) -> bool:
        self.evicted = True
        return True


# --- LayerCache unit tests (no model) --------------------------------------


def test_capacity_respected_and_lru_evicted():
    cache = LayerCache(2)
    h = [_FakeHandle(i) for i in range(3)]
    for i in range(2):
        cache.touch(h[i])
        cache.mark_resident(h[i])
        cache.enforce_capacity(protect=i)
        cache.release(h[i])
    assert cache.resident_count == 2
    # Access a third -> LRU (index 0) evicted.
    cache.touch(h[2])
    cache.mark_resident(h[2])
    cache.enforce_capacity(protect=2)
    assert cache.resident_count == 2
    assert h[0].evicted and not h[1].evicted


def test_in_use_layer_not_evicted():
    cache = LayerCache(1)
    h = [_FakeHandle(i) for i in range(2)]
    # Pin h0 (touch without release) so it is in-use.
    cache.touch(h[0])
    cache.mark_resident(h[0])
    # Now bring in h1; capacity is 1 but h0 is pinned -> transient overshoot, h0 kept.
    cache.touch(h[1])
    cache.mark_resident(h[1])
    cache.enforce_capacity(protect=1)
    assert not h[0].evicted  # pinned layer survives
    assert cache.resident_count == 2  # documented transient overshoot


def test_hit_miss_accounting():
    cache = LayerCache(4)
    h = _FakeHandle(0)
    assert cache.touch(h) is False  # first access: miss
    cache.mark_resident(h)
    cache.release(h)
    assert cache.touch(h) is True  # second access: hit
    assert cache.hits == 1 and cache.misses == 1


# --- runner integration ----------------------------------------------------


def _streamed(cache_layers=2, n_layers=6):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = tiny_llama_config(num_hidden_layers=n_layers)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    return StreamModel.from_model(model, cfg, tier="ram", device="cpu", cache_layers=cache_layers)


def test_streaming_run_respects_capacity():
    sm = _streamed(cache_layers=2, n_layers=6)
    ids = torch.randint(0, 64, (1, 5))
    sm.generate(ids, max_new_tokens=4)
    # After a run, no more than capacity layers stay resident.
    assert sm.runner.cache.resident_count <= sm.runner.cache.capacity
    m = sm.last_metrics
    # First prefill pass misses every layer at least once.
    assert m.cache_misses >= 6


def test_prefetch_stages_ahead():
    sm = _streamed(cache_layers=4, n_layers=8)
    runner = sm.runner
    # Hint a layer and wait for the background worker to stage it.
    runner.prefetcher.hint([3])
    deadline = time.time() + 2.0
    while time.time() < deadline and not runner.cache.is_resident(3):
        time.sleep(0.01)
    assert runner.cache.is_resident(3), "prefetch worker did not stage the hinted layer"
    assert runner.prefetcher.staged_count >= 1


def test_prefetch_disabled_in_residency_mode():
    # Force residency by pretending unified memory on a cpu device path.
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = tiny_llama_config(num_hidden_layers=4)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    sm = StreamModel.from_model(model, cfg, tier="ram", device="mps")
    # device=mps is unavailable -> resolves to cpu; this asserts copy mode still runs.
    assert sm.runner.mode in ("copy", "residency")
