"""Thread-safe LRU cache of materialized decoder layers (prompt §8).

The cache owns the *policy* — which layers are resident, eviction order, and
in-use refcounts — while a :class:`~streamllm.runner.LayerHandle` owns the actual
weight movement. Correctness rules enforced here:

* **No eviction of in-use layers**: a layer pinned for a forward (or an in-flight
  prefetch) is never evicted (refcount guard).
* **Capacity respected**, except a transient overshoot when every resident layer
  is pinned — we never block the foreground compute, we just exceed capacity for
  one layer and shrink back as pins release (documented downside: a brief peak
  above the planned ``cache_layers``).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from .logging_utils import get_logger

if TYPE_CHECKING:
    from .metrics import RunMetrics
    from .runner import LayerHandle

_log = get_logger("cache")


class LayerCache:
    """LRU set of resident layers with refcounting and hit/miss accounting."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(int(capacity), 1)
        self._lock = threading.RLock()
        # index -> handle, ordered oldest..newest (LRU at the front).
        self._resident: OrderedDict[int, LayerHandle] = OrderedDict()
        self._in_use: dict[int, int] = {}
        self.hits = 0
        self.misses = 0
        self._metrics: RunMetrics | None = None

    def bind_metrics(self, metrics: RunMetrics) -> None:
        self._metrics = metrics
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------- foreground

    def touch(self, handle: LayerHandle) -> bool:
        """Record access to ``handle`` and pin it. Returns True on a cache hit.

        A hit means the layer was already resident (possibly prefetched). Either
        way the layer is pinned so it cannot be evicted until :meth:`release`.
        """
        idx = handle.index
        with self._lock:
            hit = idx in self._resident
            if hit:
                self._resident.move_to_end(idx)
                self.hits += 1
            else:
                self.misses += 1
            self._in_use[idx] = self._in_use.get(idx, 0) + 1
            if self._metrics is not None:
                self._metrics.cache_hits = self.hits
                self._metrics.cache_misses = self.misses
            return hit

    def mark_resident(self, handle: LayerHandle) -> None:
        """Insert ``handle`` as the most-recently-used resident layer."""
        with self._lock:
            self._resident[handle.index] = handle
            self._resident.move_to_end(handle.index)

    def release(self, handle: LayerHandle) -> None:
        """Unpin ``handle`` after its forward completes."""
        idx = handle.index
        with self._lock:
            n = self._in_use.get(idx, 0)
            if n <= 1:
                self._in_use.pop(idx, None)
            else:
                self._in_use[idx] = n - 1

    def enforce_capacity(self, protect: int) -> list[LayerHandle]:
        """Evict LRU resident layers until at/under capacity. ``protect`` is kept.

        Skips any pinned layer. Returns the handles actually evicted (so the
        caller/handle can free device memory outside the lock if desired).
        """
        evicted: list[LayerHandle] = []
        with self._lock:
            while len(self._resident) > self.capacity:
                victim_idx = self._first_evictable(protect)
                if victim_idx is None:
                    break  # everything is pinned; tolerate transient overshoot
                handle = self._resident.pop(victim_idx)
                evicted.append(handle)
        for h in evicted:
            h.evict()
        return evicted

    def _first_evictable(self, protect: int) -> int | None:
        for idx in self._resident:  # oldest first
            if idx == protect:
                continue
            if self._in_use.get(idx, 0) > 0:
                continue
            return idx
        return None

    # -------------------------------------------------------------- prefetch

    def has_free_slot(self) -> bool:
        with self._lock:
            return len(self._resident) < self.capacity

    def is_resident(self, idx: int) -> bool:
        with self._lock:
            return idx in self._resident

    def reset(self) -> None:
        """Evict everything (e.g. between runs)."""
        with self._lock:
            handles = list(self._resident.values())
            self._resident.clear()
            self._in_use.clear()
        for h in handles:
            h.evict()

    @property
    def resident_count(self) -> int:
        with self._lock:
            return len(self._resident)
