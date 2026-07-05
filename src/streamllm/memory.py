"""Memory budget model (prompt §6) — estimate **peak**, not load-time, memory.

This is the usual failure point, so the math is explicit and every intermediate
number is exposed via :class:`MemoryEstimate` (read by ``.describe()`` and the
tiering policy). Two sources feed it:

* **analytic** — pure arithmetic from ``config.json`` (no weights, no model
  build). Used by ``streamllm describe`` so users can plan on a machine that
  can't run the model. Assumes SwiGLU-style llama/qwen decoder layers.
* **measured** — exact param counts from a meta-device skeleton, passed in via
  ``per_layer_params`` / ``resident_params`` to refine the analytic guess.

The single most important rule: KV-cache size uses ``num_key_value_heads``
(GQA/MQA), never ``num_attention_heads``. Getting that wrong is the #1 bug.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .errors import MemoryEstimationError

# Weight bytes-per-param for the supported quant schemes. int4 (NF4) stores 4
# bits + small per-block absmax scales; we approximate at 0.5 B/param and ignore
# the ~3% scale overhead (documented). int8 is 1 B/param + negligible scales.
_QUANT_BYTES_PER_PARAM = {None: None, "int8": 1.0, "int4": 0.5}


def dtype_size_bytes(dtype: object) -> float:
    """Bytes per element for a torch dtype or dtype-like string."""
    if isinstance(dtype, torch.dtype):
        return float(torch.empty(0, dtype=dtype).element_size())
    name = str(dtype).lower().replace("torch.", "")
    table = {
        "float32": 4.0,
        "fp32": 4.0,
        "float": 4.0,
        "float16": 2.0,
        "fp16": 2.0,
        "half": 2.0,
        "bfloat16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "uint8": 1.0,
        "float8": 1.0,
    }
    if name not in table:
        raise MemoryEstimationError(f"unknown dtype {dtype!r}; pass a torch.dtype or known name")
    return table[name]


def _cfg(config: object, *names: str, default: Any = None) -> Any:
    """First present attribute among ``names`` (handles HF config aliases)."""
    for n in names:
        if hasattr(config, n) and getattr(config, n) is not None:
            return getattr(config, n)
    return default


@dataclass(slots=True)
class ParamCounts:
    """Analytic parameter counts derived from a config."""

    n_layers: int
    per_layer_params: int
    resident_params: int
    is_moe: bool
    num_experts: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool


def count_params_from_config(config: object) -> ParamCounts:
    """Analytically count decoder-layer and resident params from an HF config.

    Assumes a gated-MLP (SwiGLU) llama/qwen-style layer: q/k/v/o projections +
    gate/up/down + 2 RMSNorms. MoE layers replace the single MLP with
    ``num_experts`` experts (+ router); we still count them so estimates are
    honest, even though the runner rejects MoE streaming.
    """
    hidden = int(_cfg(config, "hidden_size", "d_model", default=0) or 0)
    n_layers = int(_cfg(config, "num_hidden_layers", "n_layer", "num_layers", default=0) or 0)
    n_heads = int(_cfg(config, "num_attention_heads", "n_head", default=0) or 0)
    if hidden <= 0 or n_layers <= 0 or n_heads <= 0:
        raise MemoryEstimationError(
            "config missing hidden_size/num_hidden_layers/num_attention_heads; "
            "cannot estimate memory analytically"
        )
    n_kv = int(_cfg(config, "num_key_value_heads", default=n_heads) or n_heads)
    head_dim = int(_cfg(config, "head_dim", default=hidden // n_heads) or hidden // n_heads)
    inter = int(_cfg(config, "intermediate_size", "ffn_dim", default=4 * hidden) or 4 * hidden)
    vocab = int(_cfg(config, "vocab_size", default=0) or 0)
    tied = bool(_cfg(config, "tie_word_embeddings", default=False))
    attn_bias = bool(_cfg(config, "attention_bias", "qkv_bias", default=False))

    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    attn = hidden * q_dim + hidden * kv_dim * 2 + q_dim * hidden  # q + k + v + o
    if attn_bias:
        attn += q_dim + 2 * kv_dim + hidden  # q,k,v,o biases (approx)

    num_experts = int(_cfg(config, "num_local_experts", "num_experts", default=0) or 0)
    is_moe = num_experts > 0
    if is_moe:
        moe_inter = int(_cfg(config, "moe_intermediate_size", default=inter) or inter)
        mlp = num_experts * (3 * hidden * moe_inter) + hidden * num_experts  # experts + router
    else:
        mlp = 3 * hidden * inter  # gate + up + down

    per_layer = attn + mlp + 2 * hidden  # + input & post-attention norms

    embed = vocab * hidden
    final_norm = hidden
    lm_head = 0 if tied else vocab * hidden
    resident = embed + final_norm + lm_head

    return ParamCounts(
        n_layers=n_layers,
        per_layer_params=int(per_layer),
        resident_params=int(resident),
        is_moe=is_moe,
        num_experts=num_experts,
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        vocab_size=vocab,
        tied_embeddings=tied,
    )


@dataclass(slots=True)
class MemoryEstimate:
    """Every number in the budget model, in bytes unless noted (prompt §6)."""

    # structure
    n_layers: int
    per_layer_params: int
    resident_params: int
    is_moe: bool
    num_experts: int
    num_key_value_heads: int
    head_dim: int
    sliding_window: int | None

    # byte sizing
    weight_bytes_per_param: float
    kv_dtype_bytes: float
    per_layer_bytes: int
    resident_bytes: int
    weights_bytes: int  # all decoder layers + resident

    # context-dependent terms
    context: int
    prompt_len: int
    batch_size: int
    kv_bytes: int
    activation_bytes: int

    source: str  # "analytic" | "measured"

    # --------------------------------------------------------------- accessors

    def tier0_peak_bytes(self, overhead_reserve: int) -> int:
        """Peak if everything is resident on one device (Tier 0)."""
        return self.weights_bytes + self.kv_bytes + self.activation_bytes + overhead_reserve

    def streaming_device_peak_bytes(
        self, cache_layers: int, prefetch_buffers: int, overhead_reserve: int
    ) -> int:
        """Peak *device* memory when streaming ``cache_layers`` resident layers.

        Adds the resident set (embed/norm/head), the LRU-resident decoder layers,
        the in-flight prefetch staging buffers, KV cache, activations, overhead.
        """
        staged = max(prefetch_buffers, 0) * self.per_layer_bytes
        resident_layers = max(cache_layers, 0) * self.per_layer_bytes
        return (
            self.resident_bytes
            + resident_layers
            + staged
            + self.kv_bytes
            + self.activation_bytes
            + overhead_reserve
        )

    def as_dict(self) -> dict[str, object]:
        """Flat dict for ``.describe()`` / JSON, with GB conveniences added."""
        d = asdict(self)
        for k in (
            "per_layer_bytes",
            "resident_bytes",
            "weights_bytes",
            "kv_bytes",
            "activation_bytes",
        ):
            d[f"{k.replace('_bytes', '')}_gb"] = round(d[k] / 1e9, 3)
        return d


def estimate_memory(
    config: object,
    *,
    dtype: object = "bfloat16",
    quantize: str | None = None,
    max_context: int,
    prompt_len: int,
    batch_size: int = 1,
    activation_factor: float = 2.5,
    kv_dtype_bytes: float | None = None,
    per_layer_params: int | None = None,
    resident_params: int | None = None,
) -> MemoryEstimate:
    """Build a :class:`MemoryEstimate` (prompt §6).

    Args:
        config: An HF config (or duck-typed object with the same fields).
        dtype: Compute dtype; sets the unquantized weight + activation byte size.
        quantize: ``None`` | ``"int8"`` | ``"int4"`` — shrinks *weight* bytes only.
        max_context: ``prompt_len + max_new_tokens`` budget for KV growth.
        prompt_len: Prefill length, dominates the activation peak.
        batch_size: Sequences processed together (KV and activation scale with it).
        activation_factor: Overhead multiplier on the prefill activation estimate.
        kv_dtype_bytes: Bytes per KV element; defaults to the compute dtype size.
        per_layer_params / resident_params: Measured overrides (else analytic).
    """
    if quantize not in _QUANT_BYTES_PER_PARAM:
        raise MemoryEstimationError(f"quantize must be one of {list(_QUANT_BYTES_PER_PARAM)}")

    counts = count_params_from_config(config)
    if per_layer_params is not None:
        counts.per_layer_params = int(per_layer_params)
        source = "measured"
    else:
        source = "analytic"
    if resident_params is not None:
        counts.resident_params = int(resident_params)
        source = "measured"

    compute_bytes = dtype_size_bytes(dtype)
    weight_bpp = _QUANT_BYTES_PER_PARAM[quantize]
    if weight_bpp is None:
        weight_bpp = compute_bytes
    kv_bpp = float(kv_dtype_bytes) if kv_dtype_bytes is not None else compute_bytes

    hidden = counts.hidden_size
    n_kv = counts.num_key_value_heads
    head_dim = counts.head_dim
    sliding = _cfg(config, "sliding_window", default=None)
    sliding_window = int(sliding) if sliding else None

    per_layer_bytes = int(counts.per_layer_params * weight_bpp)
    resident_bytes = int(counts.resident_params * weight_bpp)
    weights_bytes = per_layer_bytes * counts.n_layers + resident_bytes

    kv_len = min(max_context, sliding_window) if sliding_window else max_context
    kv_bytes = int(2 * counts.n_layers * n_kv * head_dim * kv_len * batch_size * kv_bpp)

    activation_bytes = int(batch_size * prompt_len * hidden * compute_bytes * activation_factor)

    return MemoryEstimate(
        n_layers=counts.n_layers,
        per_layer_params=counts.per_layer_params,
        resident_params=counts.resident_params,
        is_moe=counts.is_moe,
        num_experts=counts.num_experts,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        sliding_window=sliding_window,
        weight_bytes_per_param=weight_bpp,
        kv_dtype_bytes=kv_bpp,
        per_layer_bytes=per_layer_bytes,
        resident_bytes=resident_bytes,
        weights_bytes=weights_bytes,
        context=max_context,
        prompt_len=prompt_len,
        batch_size=batch_size,
        kv_bytes=kv_bytes,
        activation_bytes=activation_bytes,
        source=source,
    )
