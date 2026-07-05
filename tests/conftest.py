"""Shared fixtures: tiny random-weight configs + mocked hardware budgets.

Everything here is CPU-only and network-free. Tiny configs (a few layers, hidden
64) keep the suite fast while still exercising GQA, tied embeddings, sliding
windows and two architectures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GiB = 1024**3


def tiny_llama_config(**overrides: object):
    """A tiny Llama config (GQA: 4 attn heads, 2 KV heads) for fast tests."""
    from transformers import LlamaConfig

    params: dict[str, object] = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
    )
    params.update(overrides)
    return LlamaConfig(**params)


def tiny_qwen_config(**overrides: object):
    """A tiny Qwen2 config (different module names than Llama) for discovery tests."""
    from transformers import Qwen2Config

    params: dict[str, object] = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    params.update(overrides)
    return Qwen2Config(**params)


def make_hw(
    *,
    cuda: bool = False,
    vram_total_gb: float = 0.0,
    vram_free_gb: float = 0.0,
    ram_total_gb: float = 32.0,
    ram_avail_gb: float = 16.0,
    disk_free_gb: float = 100.0,
    mps: bool = False,
    unified: bool = False,
    cache_dir: Path | None = None,
):
    """Construct a :class:`HardwareInfo` with fully mocked budgets (bytes)."""
    from streamllm.hardware import CudaDevice, HardwareInfo

    devices = []
    if cuda:
        devices.append(
            CudaDevice(
                index=0,
                name="MockGPU",
                total_bytes=int(vram_total_gb * GiB),
                free_bytes=int(vram_free_gb * GiB),
            )
        )
    return HardwareInfo(
        cuda_available=cuda,
        cuda_devices=devices,
        mps_available=mps,
        total_ram_bytes=int(ram_total_gb * GiB),
        available_ram_bytes=int(ram_avail_gb * GiB),
        disk_total_bytes=int((disk_free_gb + 50) * GiB),
        disk_free_bytes=int(disk_free_gb * GiB),
        cache_dir=cache_dir or Path("/tmp/streamllm-test-cache"),
        unified_memory=unified,
    )


@pytest.fixture
def llama_cfg():
    return tiny_llama_config()


@pytest.fixture
def qwen_cfg():
    return tiny_qwen_config()
