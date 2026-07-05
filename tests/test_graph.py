"""Module-graph discovery (prompt §15.4) — two architectures + override path."""

from __future__ import annotations

import pytest
import torch.nn as nn
from tests.conftest import tiny_llama_config, tiny_qwen_config

from streamllm.errors import GraphDiscoveryError
from streamllm.graph import discover_graph


def _meta_model(config):
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        return AutoModelForCausalLM.from_config(config)


def test_discovery_llama_no_hardcoded_names():
    cfg = tiny_llama_config()
    model = _meta_model(cfg)
    g = discover_graph(model, cfg)
    assert g.num_layers == cfg.num_hidden_layers
    assert isinstance(g.layers, nn.ModuleList)
    assert isinstance(g.embed_tokens, nn.Embedding)
    assert g.embed_tokens.num_embeddings == cfg.vocab_size
    assert isinstance(g.lm_head, nn.Linear)
    assert g.lm_head.out_features == cfg.vocab_size
    # The final norm must be OUTSIDE the decoder stack.
    assert "norm" in g.final_norm.__class__.__name__.lower()
    assert g.rotary_emb is not None  # transformers 5.x exposes rotary_emb


def test_discovery_qwen_different_arch():
    cfg = tiny_qwen_config()
    model = _meta_model(cfg)
    g = discover_graph(model, cfg)
    assert g.num_layers == cfg.num_hidden_layers
    assert g.embed_tokens.num_embeddings == cfg.vocab_size
    # Qwen tiny config has tied embeddings -> head reuses embedding weight.
    assert g.lm_head is not None


def test_discovery_picks_longest_modulelist():
    cfg = tiny_llama_config(num_hidden_layers=6)
    model = _meta_model(cfg)
    g = discover_graph(model, cfg)
    assert g.num_layers == 6
    # Path should be the decoder stack, not e.g. an attention sublist.
    assert g.layers_path.endswith("layers")


class _Unrecognized(nn.Module):
    """A model with NO ModuleList — discovery must fail loudly."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, 32)
        self.blob = nn.Linear(32, 32)
        self.head = nn.Linear(32, 64)


def test_unrecognized_structure_raises_actionable():
    model = _Unrecognized()

    class _Cfg:
        vocab_size = 64
        num_hidden_layers = 0

    with pytest.raises(GraphDiscoveryError, match="layer_module_path"):
        discover_graph(model, _Cfg())


class _CustomStack(nn.Module):
    """Non-standard names, but still has a ModuleList we can point at."""

    def __init__(self) -> None:
        super().__init__()
        self.tok = nn.Embedding(64, 32)
        self.blocks = nn.ModuleList([nn.Linear(32, 32) for _ in range(3)])
        self.final_ln = nn.LayerNorm(32)
        self.proj_out = nn.Linear(32, 64)


def test_override_path_on_custom_structure():
    model = _CustomStack()

    class _Cfg:
        vocab_size = 64
        num_hidden_layers = 3

    g = discover_graph(
        model,
        _Cfg(),
        layer_module_path="blocks",
        resident_module_paths={"embed": "tok", "norm": "final_ln", "lm_head": "proj_out"},
    )
    assert g.num_layers == 3
    assert g.embed_tokens is model.tok
    assert g.final_norm is model.final_ln
    assert g.lm_head is model.proj_out
