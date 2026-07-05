"""Exception hierarchy for streamllm.

Every error here is meant to be *actionable*: the message should say what failed,
the relevant numbers, and the flag/override that fixes it (see prompt §10).
"""

from __future__ import annotations


class StreamLLMError(Exception):
    """Base class for all streamllm errors."""


class ConfigurationError(StreamLLMError):
    """A ``StreamConfig`` value or ``from_pretrained`` argument is invalid."""


class HardwareDetectionError(StreamLLMError):
    """Hardware probing failed in a way we cannot safely recover from."""


class MemoryEstimationError(StreamLLMError):
    """The model config lacked fields needed to estimate memory."""


class TierSelectionError(StreamLLMError):
    """No tier is feasible for the requested model + hardware + context."""


class GraphDiscoveryError(StreamLLMError):
    """Could not locate the decoder stack / embedding / norm / head generically.

    Raised with guidance to pass ``layer_module_path=`` / ``resident_module_paths=``.
    """


class UnsupportedModelError(StreamLLMError):
    """The model architecture is detected but deliberately not handled (e.g. MoE)."""


class ShardError(StreamLLMError):
    """Sharding/manifest/mmap-load failure (hash mismatch, disk full, IO error)."""


class QuantizationError(StreamLLMError):
    """Quantization was requested but cannot be performed (e.g. missing bitsandbytes)."""


class ContextOverflowError(StreamLLMError):
    """``prompt_len + max_new_tokens`` exceeds the model's max positions.

    Only raised when ``on_context_overflow="error"`` (the default).
    """


class GenerationError(StreamLLMError):
    """A failure during the generation loop (bad sampling args, etc.)."""


class OutOfMemoryDemotionError(StreamLLMError):
    """The bottom tier still OOM'd after graceful demotion attempts."""
