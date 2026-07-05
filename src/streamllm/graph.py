"""Module-graph discovery (prompt §7) — find the decoder stack generically.

No hardcoded layer-name strings. We locate, for an arbitrary HF causal LM:

* the decoder ``nn.ModuleList`` (the long repeated stack),
* the input embedding (``nn.Embedding`` sized to the vocab),
* the final norm (a norm sibling of the stack, outside it),
* the LM head (``nn.Linear`` to the vocab, outside the stack), and
* the rotary-embedding module (transformers 5.x precomputes ``position_embeddings``).

Works on Llama- and Qwen-family at minimum. On an unrecognized structure we fail
with an actionable message naming the ``layer_module_path`` /
``resident_module_paths`` overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from .errors import GraphDiscoveryError
from .logging_utils import get_logger

_log = get_logger("graph")


@dataclass(slots=True)
class ModelGraph:
    """Resolved references to the streamable stack and resident modules."""

    base_model: nn.Module
    layers: nn.ModuleList
    layers_path: str
    num_layers: int
    embed_tokens: nn.Module
    final_norm: nn.Module
    lm_head: nn.Module
    rotary_emb: nn.Module | None = None
    resident_paths: dict[str, str] = field(default_factory=dict)

    def layer_module_paths(self) -> list[str]:
        return [f"{self.layers_path}.{i}" for i in range(self.num_layers)]


def _get_module(root: nn.Module, path: str) -> nn.Module:
    mod: nn.Module = root
    for part in path.split("."):
        if not part:
            continue
        if not hasattr(mod, part):
            raise GraphDiscoveryError(f"module path {path!r} not found (no attribute {part!r})")
        mod = getattr(mod, part)
    return mod


def _parent_path(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else ""


def _named_modules(root: nn.Module) -> list[tuple[str, nn.Module]]:
    return list(root.named_modules())


def _is_norm(mod: nn.Module) -> bool:
    name = mod.__class__.__name__.lower()
    return "norm" in name


def discover_graph(
    model: nn.Module,
    config: object,
    *,
    layer_module_path: str | None = None,
    resident_module_paths: dict[str, str] | None = None,
) -> ModelGraph:
    """Discover the decoder stack + resident modules of ``model``.

    Args:
        model: A causal-LM ``nn.Module`` (may be on the meta device).
        config: The model's HF config (for ``vocab_size`` / ``num_hidden_layers``).
        layer_module_path: Override for the decoder ``ModuleList`` path, e.g.
            ``"model.layers"``. Use when auto-discovery picks the wrong stack.
        resident_module_paths: Override dict with any of keys ``embed`` / ``norm`` /
            ``lm_head`` / ``rotary`` mapping to dotted module paths.

    Raises:
        GraphDiscoveryError: If the structure is unrecognized and not overridden.
    """
    overrides = resident_module_paths or {}
    vocab = int(getattr(config, "vocab_size", 0) or 0)
    n_layers_cfg = int(getattr(config, "num_hidden_layers", 0) or 0)

    # 1) The decoder stack: an explicit override, else the ModuleList whose length
    #    matches num_hidden_layers, else simply the longest ModuleList.
    if layer_module_path is not None:
        layers = _get_module(model, layer_module_path)
        if not isinstance(layers, nn.ModuleList):
            raise GraphDiscoveryError(
                f"layer_module_path={layer_module_path!r} is {type(layers).__name__}, "
                "not an nn.ModuleList"
            )
        layers_path = layer_module_path
    else:
        candidates = [
            (name, m)
            for name, m in _named_modules(model)
            if isinstance(m, nn.ModuleList) and len(m) > 0
        ]
        if not candidates:
            raise GraphDiscoveryError(
                "no nn.ModuleList found; cannot locate the decoder stack. Pass "
                "layer_module_path='...' (e.g. 'model.layers')."
            )
        exact = [c for c in candidates if len(c[1]) == n_layers_cfg] if n_layers_cfg else []
        layers_path, layers = (exact or sorted(candidates, key=lambda c: len(c[1])))[
            -1 if not exact else 0
        ]

    num_layers = len(layers)
    base_path = _parent_path(layers_path)
    base_model = _get_module(model, base_path) if base_path else model

    # Names that live *inside* the stack — excluded from resident-module search.
    layer_prefix = layers_path + "."

    def _outside_stack(name: str) -> bool:
        return not name.startswith(layer_prefix)

    named = _named_modules(model)

    # 2) Input embedding: an nn.Embedding sized to the vocab (prefer under base).
    embed = _resolve_override(model, overrides, "embed")
    if embed is None:
        embeds = [
            (name, m) for name, m in named if isinstance(m, nn.Embedding) and _outside_stack(name)
        ]
        sized = [c for c in embeds if vocab and c[1].num_embeddings == vocab]
        pool = sized or embeds
        if not pool:
            raise GraphDiscoveryError(
                "no input nn.Embedding found; pass resident_module_paths={'embed': '...'}"
            )
        embed = pool[0][1]

    # 3) LM head: an nn.Linear to the vocab, outside the stack (tie-safe).
    lm_head = _resolve_override(model, overrides, "lm_head")
    if lm_head is None:
        heads = [
            (name, m)
            for name, m in named
            if isinstance(m, nn.Linear) and _outside_stack(name) and m.out_features == vocab
        ]
        if not heads:
            # Some tied models expose no separate head module; fall back to embed.
            _log.debug("no separate lm_head Linear found; will reuse the embedding (tied)")
            lm_head = embed
        else:
            # Prefer one whose name hints at a head; else the last (top-level).
            named_hint = [h for h in heads if any(k in h[0].lower() for k in ("head", "lm", "out"))]
            lm_head = (named_hint or heads)[-1][1]

    # 4) Final norm: a norm sibling of the stack, outside it. Prefer a direct
    #    child of base_model (i.e. base_path + single segment).
    final_norm = _resolve_override(model, overrides, "norm")
    if final_norm is None:
        norms = [(name, m) for name, m in named if _is_norm(m) and _outside_stack(name)]
        if not norms:
            raise GraphDiscoveryError(
                "no final norm found outside the decoder stack; pass "
                "resident_module_paths={'norm': '...'}"
            )
        depth = (len(base_path.split(".")) + 1) if base_path else 1
        direct = [n for n in norms if len(n[0].split(".")) == depth]
        final_norm = (direct or norms)[-1][1]

    # 5) Rotary embedding (optional; transformers 5.x precomputes position_embeddings).
    rotary = _resolve_override(model, overrides, "rotary")
    if rotary is None:
        rotaries = [
            m
            for name, m in named
            if "rotary" in m.__class__.__name__.lower() or name.endswith("rotary_emb")
        ]
        rotary = rotaries[0] if rotaries else None

    graph = ModelGraph(
        base_model=base_model,
        layers=layers,
        layers_path=layers_path,
        num_layers=num_layers,
        embed_tokens=embed,
        final_norm=final_norm,
        lm_head=lm_head,
        rotary_emb=rotary,
        resident_paths={
            "layers": layers_path,
            "base": base_path or "<root>",
        },
    )
    _log.debug(
        "graph: stack=%s (%d layers) embed=%s norm=%s head=%s rotary=%s",
        layers_path,
        num_layers,
        type(embed).__name__,
        type(final_norm).__name__,
        type(lm_head).__name__,
        type(rotary).__name__ if rotary else None,
    )
    return graph


def _resolve_override(model: nn.Module, overrides: dict[str, str], key: str) -> nn.Module | None:
    path = overrides.get(key)
    if path is None:
        return None
    return _get_module(model, path)
