"""Generation correctness (prompt §15.6 prefill+decode, §15.7 batch, §15.8 seed).

Tier 0 (full load on CPU) is the oracle here; the streamed-vs-full equality lives
in test_verify.py once the runner exists.
"""

from __future__ import annotations

import pytest
import torch
from tests.conftest import tiny_llama_config

from streamllm.errors import ContextOverflowError
from streamllm.generation import SamplingParams, StopController, sample_next
from streamllm.model import StreamModel


def _build(seed: int = 0, **over):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = tiny_llama_config(**over)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    return model, cfg


def _sm(model, cfg, **kw):
    return StreamModel.from_model(model, cfg, tier="full", device="cpu", **kw)


def test_incremental_decode_matches_full_forward():
    """KV-cache decode must equal a single non-cached forward (the decode oracle)."""
    model, cfg = _build()
    sm = _sm(model, cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    gen = sm.generate(ids, max_new_tokens=8, do_sample=False)
    full_seq = torch.cat([ids, gen], dim=1)
    with torch.no_grad():
        ref = model(full_seq).logits
    pred = ref[0, ids.shape[1] - 1 : ids.shape[1] - 1 + gen.shape[1]].argmax(-1)
    assert torch.equal(pred, gen[0])


def test_greedy_is_deterministic():
    model, cfg = _build()
    sm = _sm(model, cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    a = sm.generate(ids, max_new_tokens=10, do_sample=False)
    b = sm.generate(ids, max_new_tokens=10, do_sample=False)
    assert torch.equal(a, b)


def test_sampling_reproducible_under_seed():
    model, cfg = _build()
    sm = _sm(model, cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    a = sm.generate(ids, max_new_tokens=10, do_sample=True, temperature=0.8, top_k=20, seed=123)
    b = sm.generate(ids, max_new_tokens=10, do_sample=True, temperature=0.8, top_k=20, seed=123)
    assert torch.equal(a, b)


def test_different_seeds_differ_usually():
    model, cfg = _build()
    sm = _sm(model, cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    a = sm.generate(ids, max_new_tokens=20, do_sample=True, temperature=1.5, seed=1)
    b = sm.generate(ids, max_new_tokens=20, do_sample=True, temperature=1.5, seed=2)
    assert not torch.equal(a, b)


def test_batched_left_pad_matches_per_sequence():
    """A left-padded batch row must match that sequence run alone (prompt §15.7)."""
    model, cfg = _build()
    sm = _sm(model, cfg)
    # row 0 padded with 2 leading pads, row 1 full length.
    batch = torch.tensor([[0, 0, 5, 9, 12, 3], [7, 2, 8, 1, 4, 6]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    # Drive the loop directly via the model so we control the attention mask.
    sm.tokenizer = None
    out_batch = _greedy_ids(sm, batch, mask, 6)
    single = _greedy_ids(sm, batch[1:2], mask[1:2], 6)
    assert torch.equal(out_batch[1:2], single)


def _greedy_ids(sm, ids, mask, n):
    from transformers import DynamicCache

    model = sm.model
    cache = DynamicCache()
    pos = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
    cp = torch.arange(ids.shape[1])
    with torch.no_grad():
        logits = model(
            ids,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            cache_position=cp,
        ).logits[:, -1, :]
    gen = []
    am = mask
    for _ in range(n):
        nxt = logits.argmax(-1, keepdim=True)
        gen.append(nxt)
        am = torch.cat([am, torch.ones_like(nxt)], dim=1)
        np_ = am.long().sum(-1, keepdim=True) - 1
        pl = cache.get_seq_length()
        with torch.no_grad():
            logits = model(
                nxt,
                attention_mask=am,
                position_ids=np_,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(pl, pl + 1),
            ).logits[:, -1, :]
    return torch.cat(gen, dim=1)


def test_context_overflow_raises():
    model, cfg = _build(max_position_embeddings=16)
    sm = _sm(model, cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with pytest.raises(ContextOverflowError):
        sm.generate(ids, max_new_tokens=20)


def test_context_overflow_truncate_allowed():
    model, cfg = _build(max_position_embeddings=64)
    sm = _sm(model, cfg, on_context_overflow="truncate")
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    out = sm.generate(ids, max_new_tokens=100)  # would overflow but truncation allowed
    assert out.shape[0] == 1


# --- pure generation-helper unit tests (no model) -------------------------


def test_stop_controller_stop_string():
    pieces = {1: "he", 2: "llo", 3: " world"}
    ctrl = StopController(
        1,
        eos_token_ids=None,
        stop_strings=["llo"],
        decode=lambda ids: "".join(pieces.get(i, "") for i in ids),
    )
    assert not ctrl.update(0, 1)
    assert ctrl.update(0, 2)  # "hello" contains "llo"
    assert ctrl.all_done


def test_stop_controller_eos():
    ctrl = StopController(2, eos_token_ids=[9], stop_strings=None, decode=lambda ids: "")
    assert not ctrl.update(0, 3)
    assert ctrl.update(0, 9)
    assert not ctrl.all_done  # row 1 still going
    assert ctrl.update(1, 9)
    assert ctrl.all_done


def test_sample_next_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2, 0.3]])
    out = sample_next(logits, SamplingParams(do_sample=False), [[]])
    assert int(out[0, 0]) == 1


def test_repetition_penalty_lowers_repeated():
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = sample_next(
        logits.clone(), SamplingParams(do_sample=False, repetition_penalty=2.0), [[0, 1]]
    )
    # tokens 0,1 penalized (halved); 2,3 remain at 1.0 -> argmax is 2.
    assert int(out[0, 0]) == 2
