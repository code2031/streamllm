"""Hardware detection + device resolution (prompt §4)."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_hw

from streamllm.hardware import detect_hardware


def test_detect_hardware_runs(tmp_path: Path):
    hw = detect_hardware(tmp_path / "cache")
    assert hw.total_ram_bytes > 0
    assert hw.available_ram_bytes > 0
    assert hw.disk_free_bytes >= 0
    # cache dir should have been created for the disk probe
    assert (tmp_path / "cache").exists()


def test_resolve_device_auto_prefers_cuda():
    hw = make_hw(cuda=True, vram_free_gb=8)
    assert hw.resolve_device("auto") == "cuda:0"


def test_resolve_device_auto_falls_to_cpu():
    hw = make_hw(cuda=False, mps=False)
    assert hw.resolve_device("auto") == "cpu"


def test_resolve_device_cuda_request_without_cuda_degrades():
    hw = make_hw(cuda=False)
    assert hw.resolve_device("cuda") == "cpu"


def test_unified_memory_reports_min_of_pools():
    hw = make_hw(cuda=True, vram_total_gb=128, vram_free_gb=10, ram_avail_gb=4, unified=True)
    # unified -> available is min(gpu_free=10, ram_avail=4) = 4 GiB
    assert hw.device_available_bytes("cuda:0") == 4 * (1024**3)


def test_discrete_memory_reports_gpu_free():
    hw = make_hw(cuda=True, vram_total_gb=24, vram_free_gb=20, ram_avail_gb=4, unified=False)
    assert hw.device_available_bytes("cuda:0") == 20 * (1024**3)
