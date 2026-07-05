"""Hardware detection (prompt §4).

We probe **available** memory, not just total, and never assume we own 100% of it:
``torch.cuda.mem_get_info`` per visible device, ``psutil`` for host RAM,
``shutil.disk_usage`` for the cache dir. The result is a plain dataclass so tests
can construct mocked budgets directly and the tiering policy stays pure.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import torch

from .logging_utils import get_logger

_log = get_logger("hardware")

# If a CUDA device's total memory is within this fraction of host RAM total, we
# treat it as unified memory (GB10 / Jetson / some APUs). On unified memory a
# host->device "copy" is residency management, not a real transfer, so Tier 2
# skips redundant copies. This is a heuristic, documented as such.
_UNIFIED_TOTAL_TOLERANCE = 0.20


@dataclass(slots=True)
class CudaDevice:
    """One visible CUDA device and its live memory budget (bytes)."""

    index: int
    name: str
    total_bytes: int
    free_bytes: int


@dataclass(slots=True)
class HardwareInfo:
    """A snapshot of detected accelerators + host RAM + cache-dir disk.

    All sizes are bytes. ``available_ram_bytes`` is what psutil reports as
    *available* (reclaimable), which is the number the budget model must use.
    """

    cuda_available: bool
    cuda_devices: list[CudaDevice]
    mps_available: bool
    total_ram_bytes: int
    available_ram_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    cache_dir: Path
    unified_memory: bool = field(default=False)

    # ------------------------------------------------------------------ helpers

    def resolve_device(self, device: str = "auto") -> str:
        """Resolve a ``device=`` spec to a concrete torch device string.

        ``"auto"`` prefers cuda:0, then mps, then cpu. An explicit spec is
        validated against what was actually detected.
        """
        if device == "auto":
            if self.cuda_available and self.cuda_devices:
                return f"cuda:{self.cuda_devices[0].index}"
            if self.mps_available:
                return "mps"
            return "cpu"
        if device.startswith("cuda"):
            if not self.cuda_available:
                _log.warning("device=%s requested but no CUDA detected; using cpu", device)
                return "cpu"
            return device if ":" in device else "cuda:0"
        if device == "mps" and not self.mps_available:
            _log.warning("device=mps requested but MPS unavailable; using cpu")
            return "cpu"
        return device

    def device_total_bytes(self, device: str) -> int:
        """Total memory of the accelerator backing ``device`` (RAM for cpu/mps)."""
        if device.startswith("cuda"):
            dev = self._cuda(device)
            return dev.total_bytes if dev else 0
        return self.total_ram_bytes

    def device_available_bytes(self, device: str) -> int:
        """Free memory the model may plan against for ``device``.

        For unified memory the GPU and host share one pool; we report the *min*
        of GPU-free and host-available so we don't double-count the same bytes.
        """
        if device.startswith("cuda"):
            dev = self._cuda(device)
            if dev is None:
                return 0
            if self.unified_memory:
                return min(dev.free_bytes, self.available_ram_bytes)
            return dev.free_bytes
        return self.available_ram_bytes

    def is_cuda(self, device: str) -> bool:
        return device.startswith("cuda")

    def _cuda(self, device: str) -> CudaDevice | None:
        idx = 0 if ":" not in device else int(device.split(":", 1)[1])
        for d in self.cuda_devices:
            if d.index == idx:
                return d
        return self.cuda_devices[0] if self.cuda_devices else None


def detect_hardware(cache_dir: Path, *, refresh: bool = True) -> HardwareInfo:
    """Probe the live machine. Never raises for a missing accelerator.

    Args:
        cache_dir: Directory whose free disk we report (created if missing).
        refresh: If True, query ``mem_get_info`` now (free memory is live).
    """
    cache_dir = Path(cache_dir)
    cuda_devices: list[CudaDevice] = []
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        try:
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                name = torch.cuda.get_device_name(i)
                cuda_devices.append(
                    CudaDevice(index=i, name=name, total_bytes=int(total), free_bytes=int(free))
                )
        except Exception as exc:  # pragma: no cover - hardware-specific
            _log.warning("CUDA present but mem_get_info failed (%s); treating as no CUDA", exc)
            cuda_available = False
            cuda_devices = []

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    vm = psutil.virtual_memory()
    total_ram = int(vm.total)
    avail_ram = int(vm.available)

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(cache_dir)
        disk_total, disk_free = int(usage.total), int(usage.free)
    except Exception as exc:  # pragma: no cover - fs-specific
        _log.warning("disk_usage(%s) failed (%s); reporting 0 free", cache_dir, exc)
        disk_total = disk_free = 0

    unified = mps_available
    if cuda_devices:
        gpu_total = cuda_devices[0].total_bytes
        if abs(gpu_total - total_ram) / max(total_ram, 1) < _UNIFIED_TOTAL_TOLERANCE:
            unified = True

    info = HardwareInfo(
        cuda_available=cuda_available,
        cuda_devices=cuda_devices,
        mps_available=mps_available,
        total_ram_bytes=total_ram,
        available_ram_bytes=avail_ram,
        disk_total_bytes=disk_total,
        disk_free_bytes=disk_free,
        cache_dir=cache_dir,
        unified_memory=unified,
    )
    _log.debug(
        "detected: cuda=%s devices=%s mps=%s unified=%s ram_avail=%.1fGB disk_free=%.1fGB",
        cuda_available,
        [(d.index, d.name, round(d.free_bytes / 1e9, 1)) for d in cuda_devices],
        mps_available,
        unified,
        avail_ram / 1e9,
        disk_free / 1e9,
    )
    return info
