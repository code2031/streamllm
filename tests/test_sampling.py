"""Sampling filter semantics + estimator scaling properties (prompt §11, §15.2).

Direct tests of the top-k / top-p / min-p logits warpers, plus hypothesis
property tests for the estimator's linear-scaling invariants (skipped cleanly if
hypothesis is not installed).
"""

from __future__ import annotations

import math

import torch
from tests.conftest import tiny_llama_config

from streamllm.generation import SamplingParams, _min_p, _top_k, _top_p, sample_next


def test_top_k_keeps_only_top_k():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = _top_k(logits.clone(), 2)
    # top-2 are indices 2 and 3; 0 and 1 masked to -inf.
    assert math.isinf(out[0, 0]) and out[0, 0] < 0
    assert math.isinf(out[0, 1]) and out[0, 1] < 0
    assert out[0, 2].item() == 3.0
    assert out[0, 3].item() == 4.0


def test_top_p_keeps_dominant_token():
    # token 0 has ~all the mass; nucleus at p=0.5 keeps only it.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    out = _top_p(logits.clone(), 0.5)
    assert torch.isfinite(out[0, 0])
    assert torch.isinf(out[0, 1:]).all()


def test_top_p_always_keeps_at_least_one():
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = _top_p(logits.clone(), 0.0)  # degenerate p; top token must survive
    assert torch.isfinite(out).sum() >= 1


def test_min_p_masks_low_probability():
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    out = _min_p(logits.clone(), 0.1)  # keep tokens with prob >= 0.1 * max
    assert torch.isfinite(out[0, 0])
    assert torch.isinf(out[0, 1:]).all()


def test_sampled_token_respects_top_k():
    # Over many samples, top_k=2 must never sample outside the top-2 indices.
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 0.0, 5.0, 6.0]])
    gen = torch.Generator().manual_seed(0)
    params = SamplingParams(do_sample=True, temperature=1.0, top_k=2)
    seen = set()
    for _ in range(50):
        seen.add(int(sample_next(logits.clone(), params, [[]], gen)[0, 0]))
    assert seen <= {2, 3}


# --- hypothesis property tests (skipped cleanly if hypothesis absent) ------

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    _HAS_HYPOTHESIS = False


if _HAS_HYPOTHESIS:

    def _est(**kw):
        from streamllm.memory import estimate_memory

        base = dict(dtype="float16", prompt_len=16, batch_size=1)
        base.update(kw)
        return estimate_memory(tiny_llama_config(), **base)

    @settings(max_examples=40, deadline=None)
    @given(ctx=st.integers(min_value=1, max_value=200_000))
    def test_kv_bytes_linear_in_context(ctx):
        e1 = _est(max_context=1)
        ec = _est(max_context=ctx)
        assert ec.kv_bytes == e1.kv_bytes * ctx

    @settings(max_examples=30, deadline=None)
    @given(batch=st.integers(min_value=1, max_value=64))
    def test_kv_bytes_linear_in_batch(batch):
        e1 = _est(max_context=128, batch_size=1)
        eb = _est(max_context=128, batch_size=batch)
        assert eb.kv_bytes == e1.kv_bytes * batch
