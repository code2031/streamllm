"""``StreamConfig`` — every tunable knob in one validated dataclass (prompt §14).

The defaults encode the prompt's guidance: 0.9 headroom, ~1 GB CUDA / ~0.5 GB
non-CUDA overhead floor, activation overhead factor in the 2–3 range. Env
overrides (``STREAMLLM_*``) layer on top so users can tune without code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

_GiB = 1024**3


def _default_cache_dir() -> Path:
    """Resolve the shard/cache directory, honoring ``STREAMLLM_CACHE_DIR``."""
    env = os.environ.get("STREAMLLM_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "streamllm"


@dataclass(slots=True)
class StreamConfig:
    """All numeric/behavioral tunables for memory estimation and streaming.

    Attributes:
        headroom: Usable fraction of *detected-available* memory (0<h<=1). The
            explicit guard against "fits at load, OOMs as KV grows". Default 0.9.
        activation_factor: Multiplier on the prefill activation estimate to cover
            attention scores, MLP intermediates and temporaries. ~2–3. Default 2.5.
        cuda_overhead_reserve_bytes: Fixed floor held back on CUDA for the CUDA
            context + allocator fragmentation + transient matmul buffers.
        cpu_overhead_reserve_bytes: The same floor for CPU/MPS backends.
        min_cache_layers: Minimum resident decoder layers for a streaming tier;
            must be >=2 for double-buffered prefetch.
        prefetch_buffers: Number of staging slots (>=2 enables N+1 overlap).
        io_retry_count: Bounded retries for a failing shard read before giving up.
        cache_dir: Where disk shards live / are written.
        kv_dtype_bytes: Bytes per KV element; ``None`` means "same as compute dtype".
        max_context_default: Fallback ``prompt+new`` budget when the user gives none
            and the model config has no usable ``max_position_embeddings``.
    """

    headroom: float = 0.9
    activation_factor: float = 2.5
    cuda_overhead_reserve_bytes: int = 1 * _GiB
    cpu_overhead_reserve_bytes: int = _GiB // 2
    min_cache_layers: int = 2
    prefetch_buffers: int = 2
    io_retry_count: int = 3
    cache_dir: Path = field(default_factory=_default_cache_dir)
    kv_dtype_bytes: int | None = None
    max_context_default: int = 4096

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ConfigurationError` on any out-of-range value."""
        if not (0.0 < self.headroom <= 1.0):
            raise ConfigurationError(
                f"headroom must be in (0, 1]; got {self.headroom}. "
                "Lower it (e.g. 0.8) if you OOM despite a tier choice."
            )
        if self.activation_factor < 1.0:
            raise ConfigurationError(
                f"activation_factor must be >=1.0; got {self.activation_factor}"
            )
        if self.min_cache_layers < 2:
            raise ConfigurationError(
                "min_cache_layers must be >=2 for double-buffered prefetch; "
                f"got {self.min_cache_layers}"
            )
        if self.prefetch_buffers < 2:
            raise ConfigurationError(
                f"prefetch_buffers must be >=2 for N+1 overlap; got {self.prefetch_buffers}"
            )
        if self.io_retry_count < 0:
            raise ConfigurationError(f"io_retry_count must be >=0; got {self.io_retry_count}")
        if self.kv_dtype_bytes is not None and self.kv_dtype_bytes <= 0:
            raise ConfigurationError(
                f"kv_dtype_bytes must be >0 or None; got {self.kv_dtype_bytes}"
            )

    def overhead_reserve_bytes(self, is_cuda: bool) -> int:
        """Return the framework-overhead floor for the active backend."""
        return self.cuda_overhead_reserve_bytes if is_cuda else self.cpu_overhead_reserve_bytes

    @classmethod
    def from_env(cls, base: StreamConfig | None = None) -> StreamConfig:
        """Apply ``STREAMLLM_*`` env overrides on top of ``base`` (or defaults)."""
        cfg = base or cls()
        overrides: dict[str, Any] = {}
        if (v := os.environ.get("STREAMLLM_HEADROOM")) is not None:
            overrides["headroom"] = float(v)
        if (v := os.environ.get("STREAMLLM_ACTIVATION_FACTOR")) is not None:
            overrides["activation_factor"] = float(v)
        if (v := os.environ.get("STREAMLLM_CACHE_DIR")) is not None:
            overrides["cache_dir"] = Path(v).expanduser()
        if (v := os.environ.get("STREAMLLM_IO_RETRY")) is not None:
            overrides["io_retry_count"] = int(v)
        if not overrides:
            return cfg
        return replace(cfg, **overrides)
