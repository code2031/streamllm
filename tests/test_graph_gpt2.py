"""Module-graph discovery on a third architecture (GPT-2) via auto-discovery.

GPT-2 has different names (transformer.wte / transformer.h / transformer.ln_f /
lm_head), learned positional embeddings (wpe) instead of RoPE, and Conv1D
attention. Discovery must still find the stack, embed, norm and head with NO
overrides, proving §7's "no hardcoded names" claim beyond Llama/Qwen.
"""

from __future__ import annotations

import torch.nn as nn


def _meta_gpt2():
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, GPT2Config

    cfg = GPT2Config(vocab_size=128, n_embd=32, n_layer=3, n_head=4, n_positions=64)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    return model, cfg


def test_gpt2_discovery_no_overrides():
    from streamllm.graph import discover_graph

    model, cfg = _meta_gpt2()
    g = discover_graph(model, cfg)
    assert g.layers_path == "transformer.h"
    assert g.num_layers == 3
    # wte (vocab-sized) is picked over wpe (position-sized).
    assert isinstance(g.embed_tokens, nn.Embedding)
    assert g.embed_tokens.num_embeddings == cfg.vocab_size
    # ln_f (LayerNorm) is the final norm, not the per-block ln_1/ln_2 inside h.
    assert "norm" in g.final_norm.__class__.__name__.lower()
    assert isinstance(g.lm_head, nn.Linear)
    assert g.lm_head.out_features == cfg.vocab_size
    # GPT-2 has no rotary embedding.
    assert g.rotary_emb is None


def test_gpt2_final_norm_is_outside_stack():
    from streamllm.graph import discover_graph

    model, cfg = _meta_gpt2()
    g = discover_graph(model, cfg)
    # The final norm must not be one of the per-layer norms inside transformer.h.
    for layer in g.layers:
        for sub in layer.modules():
            assert sub is not g.final_norm
