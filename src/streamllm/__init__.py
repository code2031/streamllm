"""streamllm — speed-first, auto-tiering LLM inference (see README / SPEC.md).

Importing the core library must not require ``bitsandbytes`` (quant is optional),
so the heavy :class:`StreamModel` is loaded lazily via module ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .config import StreamConfig
from .errors import (
    ConfigurationError,
    ContextOverflowError,
    GraphDiscoveryError,
    MemoryEstimationError,
    OutOfMemoryDemotionError,
    QuantizationError,
    ShardError,
    StreamLLMError,
    TierSelectionError,
    UnsupportedModelError,
)
from .hardware import CudaDevice, HardwareInfo, detect_hardware
from .memory import MemoryEstimate, count_params_from_config, estimate_memory
from .tiering import TIER_NAMES, TierDecision, select_tier

__version__ = "0.1.0"

if TYPE_CHECKING:
    from .model import AutoModel, StreamModel
    from .shard import shard_model

_LAZY = {
    "StreamModel": ("streamllm.model", "StreamModel"),
    "AutoModel": ("streamllm.model", "AutoModel"),
    "estimate_only": ("streamllm.model", "estimate_only"),
    "shard_model": ("streamllm.shard", "shard_model"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve heavy symbols (model/shard) on first access (PEP 562)."""
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        return getattr(import_module(mod_name), attr)
    raise AttributeError(f"module 'streamllm' has no attribute {name!r}")


__all__ = [
    "TIER_NAMES",
    "AutoModel",
    "ConfigurationError",
    "ContextOverflowError",
    "CudaDevice",
    "GraphDiscoveryError",
    "HardwareInfo",
    "MemoryEstimate",
    "MemoryEstimationError",
    "OutOfMemoryDemotionError",
    "QuantizationError",
    "ShardError",
    "StreamConfig",
    "StreamLLMError",
    "StreamModel",
    "TierDecision",
    "TierSelectionError",
    "UnsupportedModelError",
    "__version__",
    "count_params_from_config",
    "detect_hardware",
    "estimate_memory",
    "estimate_only",
    "select_tier",
    "shard_model",
]
