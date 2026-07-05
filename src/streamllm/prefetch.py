"""Background prefetcher (prompt §8) — stage layer N+1 while layer N computes.

The prefetch win is **not** Python-level parallelism (the GIL serializes Python).
It comes from the staging copy running on a dedicated CUDA stream / inside the
``.to(device, non_blocking=True)`` C call, which releases the GIL — so the H2D
transfer overlaps the compute kernels of the current layer. On CPU/unified there
is no real copy to overlap, so the worker is a no-op (``residency`` mode) or just
warms the next layer's clone for the test path.

Deadlock avoidance: the worker only stages into a *free* cache slot and never
holds the cache lock and a handle lock simultaneously (see :mod:`streamllm.cache`
and :class:`~streamllm.runner.LayerHandle`).
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

from .logging_utils import get_logger

if TYPE_CHECKING:
    from .cache import LayerCache
    from .runner import LayerHandle

_log = get_logger("prefetch")

_STOP = object()


class Prefetcher:
    """A single daemon worker that materializes hinted layers ahead of compute."""

    def __init__(self, handles: list[LayerHandle], cache: LayerCache, *, enabled: bool) -> None:
        self._handles = handles
        self._cache = cache
        self._enabled = enabled
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.staged_count = 0  # observable for the overlap smoke test

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="streamllm-prefetch", daemon=True)
        self._thread.start()

    def hint(self, indices: list[int]) -> None:
        """Schedule ``indices`` for background staging (best-effort)."""
        if not self._enabled:
            return
        for i in indices:
            if 0 <= i < len(self._handles):
                self._q.put(i)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _STOP:
                break
            idx = int(item)
            handle = self._handles[idx]
            if self._cache.is_resident(idx):
                continue
            if not self._cache.has_free_slot():
                continue  # don't overfill; the foreground will fetch it on demand
            if handle.stage():  # claims + materializes only if currently absent
                self._cache.mark_resident(handle)
                self._cache.enforce_capacity(protect=idx)
                self.staged_count += 1

    def stop(self) -> None:
        self._stop.set()
        self._q.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
