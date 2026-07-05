"""Weight-only quantization for shards + resident footprint (prompt §9).

Quantizing **at shard time** is what shrinks on-disk bytes (fewer bytes read per
layer = the main Tier-3 speedup). We quantize *weights only* (never activations),
per output-channel where it matters for accuracy, and dequantize per layer at use.

Schemes:
* ``int8`` — per-output-channel symmetric, 1 B/param. Pure torch, no extra deps.
* ``int4`` — per-output-channel symmetric, nibble-packed, ~0.5 B/param. Pure torch.
* ``nf4`` — bitsandbytes NF4 (optional dep). Requested without it → clear error.

Only 2-D floating weights are quantized; 1-D tensors (norms, biases) and the
embedding stay full precision. The accuracy/speed trade is documented in
docs/architecture.md; int4 ≈ 4× fewer bytes/param than fp16 ≈ roughly
proportional Tier-3 speedup, modulo dequant cost.
"""

from __future__ import annotations

from typing import Any

import torch

from .errors import QuantizationError

_SCALE_SUFFIX = "__scale"


def _dtype_from_str(name: str) -> torch.dtype:
    table = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
    }
    return table.get(name, torch.float32)


def _quantizable(t: torch.Tensor) -> bool:
    return t.is_floating_point() and t.dim() == 2 and min(t.shape) > 1


def _quant_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8).float()
    q = torch.round(w.float() / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def _quant_int4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (w.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-8).float()
    q = torch.round(w.float() / scale).clamp(-7, 7).to(torch.int8)
    qz = (q + 8).to(torch.uint8).reshape(-1)
    if qz.numel() % 2:
        qz = torch.cat([qz, torch.zeros(1, dtype=torch.uint8)])
    packed = (qz[0::2] | (qz[1::2] << 4)).to(torch.uint8)
    return packed, scale


def _dequant_int4(packed: torch.Tensor, scale: torch.Tensor, shape: list[int]) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    flat = torch.stack([lo, hi], dim=1).reshape(-1)[: shape[0] * shape[1]]
    q = flat.to(torch.int16) - 8
    return q.reshape(shape).float() * scale


def maybe_quantize_state(
    tensors: dict[str, torch.Tensor], scheme: str | None
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    """Quantize a layer's state dict. Returns (tensors+scales, quant_meta|None)."""
    if scheme is None:
        return tensors, None
    if scheme == "nf4":
        return _quantize_nf4(tensors)
    if scheme not in ("int8", "int4"):
        raise QuantizationError(f"unknown quantize scheme {scheme!r}")

    out: dict[str, torch.Tensor] = {}
    params_meta: dict[str, Any] = {}
    for name, t in tensors.items():
        if not _quantizable(t):
            out[name] = t
            continue
        if scheme == "int8":
            q, scale = _quant_int8(t)
            out[name] = q
            params_meta[name] = {"scheme": "int8", "orig_dtype": str(t.dtype)}
        else:
            packed, scale = _quant_int4(t)
            out[name] = packed
            params_meta[name] = {
                "scheme": "int4",
                "orig_dtype": str(t.dtype),
                "shape": list(t.shape),
            }
        out[name + _SCALE_SUFFIX] = scale
    return out, {"scheme": scheme, "params": params_meta}


def dequantize_tensor(
    t: torch.Tensor, handle: Any, name: str, quant_meta: dict[str, Any] | None
) -> torch.Tensor:
    """Reverse :func:`maybe_quantize_state` for a single fetched tensor."""
    if not quant_meta:
        return t
    pm = quant_meta.get("params", {}).get(name)
    if pm is None:
        return t
    scale = handle.get_tensor(name + _SCALE_SUFFIX)
    if pm["scheme"] == "int8":
        w = t.to(torch.float32) * scale
    elif pm["scheme"] == "int4":
        w = _dequant_int4(t, scale, pm["shape"])
    elif pm["scheme"] == "nf4":
        return _dequantize_nf4(t, handle, name, pm)
    else:  # pragma: no cover
        raise QuantizationError(f"unknown stored scheme {pm['scheme']!r}")
    return w.to(_dtype_from_str(pm["orig_dtype"]))


# --- nf4 (bitsandbytes optional dep) ---------------------------------------


def _require_bitsandbytes() -> Any:
    try:
        import bitsandbytes as bnb

        return bnb
    except ImportError as exc:  # pragma: no cover - depends on env
        raise QuantizationError(
            "quantize='nf4' needs bitsandbytes. Install it with "
            "`pip install streamllm[quant]`, or use the pure-torch 'int4'/'int8' schemes."
        ) from exc


def _quantize_nf4(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    _require_bitsandbytes()
    from bitsandbytes.functional import quantize_nf4  # type: ignore

    out: dict[str, torch.Tensor] = {}
    params_meta: dict[str, Any] = {}
    for name, t in tensors.items():
        if not _quantizable(t):
            out[name] = t
            continue
        q, state = quantize_nf4(t.to(torch.float16))
        out[name] = q
        out[name + "__absmax"] = state.absmax
        params_meta[name] = {"scheme": "nf4", "shape": list(t.shape), "orig_dtype": str(t.dtype)}
    return out, {"scheme": "nf4", "params": params_meta}


def _dequantize_nf4(  # pragma: no cover
    t: torch.Tensor, handle: Any, name: str, pm: dict
) -> torch.Tensor:
    _require_bitsandbytes()
    from bitsandbytes.functional import QuantState, dequantize_nf4

    absmax = handle.get_tensor(name + "__absmax")
    state = QuantState(
        absmax=absmax,
        shape=torch.Size(pm["shape"]),
        dtype=torch.float16,
        blocksize=64,
        quant_type="nf4",
    )
    return dequantize_nf4(t, state).to(_dtype_from_str(pm["orig_dtype"]))


def bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False
