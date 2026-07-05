"""Benchmark core (prompt §16) — honest by design.

Warmup runs are discarded; multiple timed trials report **median and p90**, not
just mean. Tokens/sec, time-to-first-token, and decode tokens/sec are reported
separately from prefill. The one-line verdict states whether the wall time is in
layer staging (I/O-bound) or compute, and the Tier-3 batch sweep demonstrates
read amortization. Output is JSON + CSV with enough metadata to reproduce.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_utils import get_logger
from .metrics import RunMetrics

_log = get_logger("benchmark")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _median(values: list[float]) -> float:
    return _percentile(values, 0.5)


@dataclass(slots=True)
class BenchResult:
    """Aggregated benchmark numbers for one model/config."""

    metadata: dict[str, Any]
    tier: int
    deciding_numbers: dict[str, float]
    tokens_per_s_median: float
    tokens_per_s_p90: float
    ttft_s_median: float
    decode_tokens_per_s_median: float
    prefill_s_median: float
    layer_load_s_mean: float
    layer_compute_s_mean: float
    cache_hit_rate: float
    io_fraction: float
    peak_vram_bytes: int
    peak_ram_bytes: int
    effective_read_gbps: float
    verdict: str
    trials: list[dict[str, Any]] = field(default_factory=list)
    batch_sweep: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]
        return d

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def write_csv(self, path: str | Path) -> None:
        flat = {k: v for k, v in self.to_dict().items() if not isinstance(v, (list, dict))}
        flat.update({f"meta.{k}": v for k, v in self.metadata.items()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            w.writeheader()
            w.writerow(flat)


def _verdict(tier: int, io_fraction: float, eff_gbps: float) -> str:
    if tier == 0:
        return "Tier 0, compute-bound: the model fits; a dedicated engine would be faster."
    if io_fraction >= 0.5:
        return (
            f"Tier {tier}, I/O-bound: {io_fraction * 100:.0f}% of layer wall time in staging; "
            f"effective read ~= {eff_gbps:.2f} GB/s (the ceiling). Quantize/batch to push it."
        )
    return f"Tier {tier}, compute-bound: only {io_fraction * 100:.0f}% of layer wall in staging."


def benchmark_model(
    sm: Any,
    prompt: Any,
    *,
    max_new_tokens: int = 32,
    trials: int = 3,
    warmup: int = 1,
    batch_sizes: list[int] | None = None,
) -> BenchResult:
    """Run warmup + timed trials against an already-built :class:`StreamModel`."""
    for _ in range(max(warmup, 0)):
        sm.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)

    runs: list[RunMetrics] = []
    trial_dicts: list[dict[str, Any]] = []
    for _ in range(max(trials, 1)):
        sm.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        m = sm.last_metrics
        runs.append(m)
        trial_dicts.append(m.as_dict())

    tps = [m.tokens_per_s for m in runs]
    ttft = [m.ttft_s or 0.0 for m in runs]
    dtps = [m.decode_tokens_per_s for m in runs]
    prefill = [m.prefill_s for m in runs]
    load = [m.layer_load_s for m in runs]
    compute = [m.layer_compute_s for m in runs]
    total_read = sum(m.bytes_read for m in runs)
    total_load = sum(m.layer_load_s for m in runs)
    eff_gbps = (total_read / total_load / 1e9) if total_load > 0 else 0.0
    io_frac = _median([m.io_fraction for m in runs])

    decision = sm.decision
    sweep = _batch_sweep(sm, prompt, max_new_tokens, batch_sizes) if batch_sizes else []

    return BenchResult(
        metadata={
            "model": sm.describe().get("model"),
            "tier": decision.tier,
            "tier_name": decision.name,
            "device": sm.device,
            "dtype": str(getattr(sm.config, "torch_dtype", None) or "default"),
            "quantization": sm.estimate.weight_bytes_per_param,
            "max_new_tokens": max_new_tokens,
            "trials": trials,
            "warmup": warmup,
            "streamllm_version": _version(),
        },
        tier=decision.tier,
        deciding_numbers={
            k: round(v / 1e9, 3)
            for k, v in decision.numbers.items()
            if k not in ("headroom", "cache_layers")
        },
        tokens_per_s_median=round(_median(tps), 3),
        tokens_per_s_p90=round(_percentile(tps, 0.9), 3),
        ttft_s_median=round(_median(ttft), 5),
        decode_tokens_per_s_median=round(_median(dtps), 3),
        prefill_s_median=round(_median(prefill), 5),
        layer_load_s_mean=round(sum(load) / len(load), 5),
        layer_compute_s_mean=round(sum(compute) / len(compute), 5),
        cache_hit_rate=round(_median([m.cache_hit_rate for m in runs]), 3),
        io_fraction=round(io_frac, 3),
        peak_vram_bytes=max((m.peak_vram_bytes for m in runs), default=0),
        peak_ram_bytes=max((m.peak_ram_bytes for m in runs), default=0),
        effective_read_gbps=round(eff_gbps, 3),
        verdict=_verdict(decision.tier, io_frac, eff_gbps),
        trials=trial_dicts,
        batch_sweep=sweep,
    )


def _batch_sweep(
    sm: Any, prompt: Any, max_new_tokens: int, batch_sizes: list[int]
) -> list[dict[str, Any]]:
    """Tokens/sec vs batch size — demonstrates Tier-3 read amortization (prompt §9)."""
    out: list[dict[str, Any]] = []
    for b in batch_sizes:
        batch = _replicate(prompt, b)
        sm.generate(batch, max_new_tokens=max_new_tokens, do_sample=False)
        m = sm.last_metrics
        out.append(
            {
                "batch_size": b,
                "total_generated": m.generated_tokens,
                "total_s": round(m.total_s, 4),
                "tokens_per_s": round(m.tokens_per_s, 3),
                "cache_hit_rate": round(m.cache_hit_rate, 3),
            }
        )
    return out


def _replicate(prompt: Any, b: int) -> Any:
    import torch

    if isinstance(prompt, str):
        return [prompt] * b
    if isinstance(prompt, list):
        return (prompt * b)[: max(b, 1)] if prompt else prompt
    if isinstance(prompt, torch.Tensor):
        ids = prompt if prompt.dim() == 2 else prompt.unsqueeze(0)
        return ids.repeat(b, 1)
    return prompt


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "unknown"
