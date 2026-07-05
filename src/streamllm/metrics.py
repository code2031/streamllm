"""Lightweight metrics the runner fills and ``.describe()`` / the benchmark read.

Kept dependency-free and trivially serializable. Per-layer load-vs-compute timing
and cache hit rate are what let the benchmark issue an honest "I/O-bound vs
compute-bound" verdict (prompt §14/§16).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class RunMetrics:
    """Counters/timers for one generation run."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_s: float | None = None  # time to first token
    prefill_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0

    layer_load_s: float = 0.0  # cumulative time spent staging/materializing layers
    layer_compute_s: float = 0.0  # cumulative time spent in layer forward
    cache_hits: int = 0
    cache_misses: int = 0

    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    bytes_read: int = 0  # total bytes streamed (disk/ram) — for I/O ceiling math

    _t0: float = field(default=0.0, repr=False)

    # ----------------------------------------------------------- timer helpers

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def mark_first_token(self) -> None:
        if self.ttft_s is None and self._t0:
            self.ttft_s = time.perf_counter() - self._t0

    def finish(self) -> None:
        if self._t0:
            self.total_s = time.perf_counter() - self._t0

    # ----------------------------------------------------------------- derived

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    @property
    def decode_tokens_per_s(self) -> float:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.generated_tokens / self.total_s if self.total_s > 0 else 0.0

    @property
    def io_fraction(self) -> float:
        """Share of layer wall-time spent loading vs computing — the verdict input."""
        wall = self.layer_load_s + self.layer_compute_s
        return self.layer_load_s / wall if wall > 0 else 0.0

    def bottleneck(self) -> str:
        """Honest one-word verdict from the load/compute split."""
        if self.layer_load_s + self.layer_compute_s == 0:
            return "unknown"
        return "io-bound" if self.io_fraction >= 0.5 else "compute-bound"

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_s": self.ttft_s,
            "prefill_s": self.prefill_s,
            "decode_s": self.decode_s,
            "total_s": self.total_s,
            "tokens_per_s": round(self.tokens_per_s, 3),
            "decode_tokens_per_s": round(self.decode_tokens_per_s, 3),
            "layer_load_s": round(self.layer_load_s, 4),
            "layer_compute_s": round(self.layer_compute_s, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "io_fraction": round(self.io_fraction, 3),
            "bottleneck": self.bottleneck(),
            "peak_vram_bytes": self.peak_vram_bytes,
            "peak_ram_bytes": self.peak_ram_bytes,
            "bytes_read": self.bytes_read,
        }
