"""Streaming runner (prompt §8) — swap decoder-layer weights via forward hooks.

We never reimplement the model's forward (that is the #1 silent-wrongness risk).
Instead each decoder layer gets a ``forward_pre_hook`` that materializes its
weights on the compute device (cache hit or stage) and a ``forward_hook`` that
releases the pin so the LRU may evict it. The model's own mask/RoPE/cache math
runs untouched, which is what lets ``verify_against_full_load`` pass.

Transfer modes:
* ``cuda``     — discrete GPU: pinned-CPU masters → async H2D on a dedicated
  stream, gated into the compute stream by a CUDA event (no use-before-ready).
* ``copy``     — CPU/MPS test path: a real ``clone`` per materialize so eviction
  frees memory and the LRU/prefetch path is genuinely exercised.
* ``residency``— unified memory / MPS: weights already live in the one pool, so
  materialize/evict are no-ops (residency management, not a real copy).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from .cache import LayerCache
from .logging_utils import get_logger
from .prefetch import Prefetcher

if TYPE_CHECKING:
    from .config import StreamConfig
    from .graph import ModelGraph
    from .memory import MemoryEstimate
    from .metrics import RunMetrics
    from .tiering import TierDecision

_log = get_logger("runner")


def _select_mode(decision: TierDecision, compute_device: str, unified: bool) -> str:
    """Pick the transfer mode. ``unified`` only matters for a CUDA device — a CPU
    compute device never touches the GPU pool, so it always uses real ``copy``."""
    if compute_device.startswith("cuda"):
        return "residency" if unified else "cuda"
    if compute_device == "mps":
        return "residency"
    return "copy"


class LayerHandle:
    """Owns weight movement + readiness for one decoder layer.

    State machine (guarded by ``_lock``): ``absent`` → ``staging`` → ``ready``.
    Only one claimant transitions ``absent``→``staging``; others wait on the
    readiness event. ``_lock`` is never held during the (slow) copy itself, so the
    cache lock and this lock are never nested.
    """

    def __init__(
        self,
        index: int,
        params: list[torch.nn.Parameter],
        fetch: Callable[[], list[torch.Tensor]],
        *,
        mode: str,
        compute_device: str,
        prefetch_stream: object | None,
        nbytes: int,
    ) -> None:
        self.index = index
        self.params = params
        self._fetch = fetch
        self.mode = mode
        self.device = compute_device
        self._stream = prefetch_stream
        self.nbytes = nbytes
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._state = "absent" if mode != "residency" else "ready"
        self._cuda_event: object | None = None
        self._device_tensors: list[torch.Tensor] = []
        if mode == "residency":
            self._ready.set()

    @property
    def state(self) -> str:
        return self._state

    def acquire_for_compute(self, compute_stream: object | None) -> None:
        """Block until this layer's weights are resident + copy-complete."""
        if self.mode == "residency":
            return
        claimed = False
        wait_event = self._ready
        with self._lock:
            if self._state == "ready":
                pass
            elif self._state == "staging":
                wait_event = self._ready
            else:
                self._state = "staging"
                claimed = True
        if claimed:
            self._materialize()
            with self._lock:
                self._state = "ready"
                self._ready.set()
        else:
            wait_event.wait()
        # Order the compute stream AFTER the copy completed (CUDA events, not sleeps).
        if self._cuda_event is not None and compute_stream is not None:
            compute_stream.wait_event(self._cuda_event)  # type: ignore[attr-defined]

    def stage(self) -> bool:
        """Background materialize. Returns True only if it claimed an absent layer."""
        if self.mode == "residency":
            return False
        with self._lock:
            if self._state != "absent":
                return False
            self._state = "staging"
        self._materialize()
        with self._lock:
            self._state = "ready"
            self._ready.set()
        return True

    def evict(self) -> bool:
        """Free the device copy if not in use. Residency mode never frees."""
        if self.mode == "residency":
            return True
        with self._lock:
            if self._state == "absent":
                return True
            self._free_device()
            self._state = "absent"
            self._ready = threading.Event()
            return True

    # ----------------------------------------------------------- internals

    def _materialize(self) -> None:
        srcs = self._fetch()
        if self.mode == "cuda" and self._stream is not None:
            dev_tensors: list[torch.Tensor] = []
            with torch.cuda.stream(self._stream):  # type: ignore[arg-type]
                for p, s in zip(self.params, srcs, strict=True):
                    t = s.to(self.device, non_blocking=True)
                    p.data = t
                    dev_tensors.append(t)
                ev = torch.cuda.Event()
                ev.record(self._stream)  # type: ignore[arg-type]
            self._cuda_event = ev
            self._device_tensors = dev_tensors
        else:  # "copy"
            dev_tensors = []
            for p, s in zip(self.params, srcs, strict=True):
                t = s.to(self.device)
                if t.data_ptr() == s.data_ptr():
                    t = s.clone()  # force a distinct copy so evict() frees memory
                p.data = t
                dev_tensors.append(t)
            self._device_tensors = dev_tensors

    def _free_device(self) -> None:
        placeholder = torch.empty(0)
        for p in self.params:
            p.data = placeholder
        self._device_tensors = []
        self._cuda_event = None


class StreamingRunner:
    """Installs streaming hooks on a model's decoder stack and drives the cache."""

    def __init__(
        self,
        model: torch.nn.Module,
        graph: ModelGraph,
        decision: TierDecision,
        cfg: StreamConfig,
        compute_device: str,
        est: MemoryEstimate,
        *,
        unified: bool = False,
        layer_fetchers: list[Callable[[], list[torch.Tensor]]] | None = None,
    ) -> None:
        self.model = model
        self.graph = graph
        self.decision = decision
        self.cfg = cfg
        self.device = compute_device
        self.est = est
        self.mode = _select_mode(decision, compute_device, unified)
        self.ahead = cfg.prefetch_buffers
        self._metrics: RunMetrics | None = None
        self._hook_handles: list[object] = []
        self._compute_start: dict[int, float] = {}

        self._prefetch_stream = (
            torch.cuda.Stream(device=compute_device) if self.mode == "cuda" else None
        )
        self.handles = self._build_handles(layer_fetchers)
        # Residency keeps everything resident; capacity = all layers.
        capacity = (
            graph.num_layers
            if self.mode == "residency"
            else max(int(decision.cache_layers or cfg.min_cache_layers), cfg.min_cache_layers)
        )
        self.cache = LayerCache(capacity)
        self.prefetcher = Prefetcher(self.handles, self.cache, enabled=(self.mode != "residency"))

    # ------------------------------------------------------------- placement

    def _build_handles(
        self, layer_fetchers: list[Callable[[], list[torch.Tensor]]] | None
    ) -> list[LayerHandle]:
        device = self.device
        # Move the resident set (embed/norm/head/rotary) onto the compute device.
        for mod in (self.graph.embed_tokens, self.graph.final_norm, self.graph.lm_head):
            mod.to(device)
        if self.graph.rotary_emb is not None:
            self.graph.rotary_emb.to(device)

        handles: list[LayerHandle] = []
        for i, layer in enumerate(self.graph.layers):
            params = list(layer.parameters())
            if self.mode == "residency":
                layer.to(device)
                masters = [p.data for p in params]
                fetch = layer_fetchers[i] if layer_fetchers else (lambda m=masters: m)
            elif layer_fetchers is not None:
                # Disk tier: weights come from a loader, not RAM. Free param storage.
                for p in params:
                    p.data = torch.empty(0, dtype=p.dtype)
                for b in layer.buffers():
                    b.data = b.data.to(device)
                fetch = layer_fetchers[i]
            else:
                masters = []
                for p in params:
                    m = p.data.detach().to("cpu")
                    if self.mode == "cuda":
                        with contextlib.suppress(RuntimeError, AssertionError):
                            m = m.pin_memory()
                    masters.append(m)
                    p.data = torch.empty(0, dtype=p.dtype)
                for b in layer.buffers():
                    b.data = b.data.to(device)
                fetch = lambda m=masters: m  # noqa: E731
            handles.append(
                LayerHandle(
                    i,
                    params,
                    fetch,
                    mode=self.mode,
                    compute_device=device,
                    prefetch_stream=self._prefetch_stream,
                    nbytes=self.est.per_layer_bytes,
                )
            )
        return handles

    def install(self) -> None:
        """Register pre/post forward hooks on every decoder layer."""
        idx_of: dict[int, int] = {id(layer): i for i, layer in enumerate(self.graph.layers)}

        def pre_hook(module: torch.nn.Module, args: object) -> None:
            idx = idx_of[id(module)]
            handle = self.handles[idx]
            t0 = time.perf_counter()
            hit = self.cache.touch(handle)
            stream = torch.cuda.current_stream(self.device) if self.mode == "cuda" else None
            handle.acquire_for_compute(stream)
            self.cache.mark_resident(handle)
            self.cache.enforce_capacity(protect=idx)
            if self._metrics is not None:
                self._metrics.layer_load_s += time.perf_counter() - t0
                if not hit:
                    self._metrics.bytes_read += handle.nbytes
            self._compute_start[idx] = time.perf_counter()
            # Stage the next few layers ahead while this one computes.
            self.prefetcher.hint([idx + k for k in range(1, self.ahead + 1)])

        def post_hook(module: torch.nn.Module, args: object, output: object) -> None:
            idx = idx_of[id(module)]
            if self._metrics is not None and idx in self._compute_start:
                self._metrics.layer_compute_s += time.perf_counter() - self._compute_start[idx]
            self.cache.release(self.handles[idx])

        for layer in self.graph.layers:
            self._hook_handles.append(layer.register_forward_pre_hook(pre_hook))
            self._hook_handles.append(layer.register_forward_hook(post_hook))
        self.prefetcher.start()
        _log.info(
            "streaming runner installed: mode=%s capacity=%d layers=%d prefetch_ahead=%d",
            self.mode,
            self.cache.capacity,
            self.graph.num_layers,
            self.ahead,
        )

    # ---------------------------------------------------------------- metrics

    def bind_metrics(self, metrics: RunMetrics) -> None:
        self._metrics = metrics
        self.cache.bind_metrics(metrics)

    def collect_peak(self, metrics: RunMetrics) -> None:
        if self.mode == "cuda":
            with contextlib.suppress(Exception):  # pragma: no cover
                metrics.peak_vram_bytes = int(torch.cuda.max_memory_allocated(self.device))
        with contextlib.suppress(Exception):  # pragma: no cover
            import psutil

            metrics.peak_ram_bytes = int(psutil.Process().memory_info().rss)

    def close(self) -> None:
        self.prefetcher.stop()
        for h in self._hook_handles:
            h.remove()  # type: ignore[attr-defined]
        self._hook_handles.clear()
