"""Tier selection policy (prompt §5/§6) — the core feature.

Given detected hardware + a :class:`MemoryEstimate` + config, choose the
*fastest viable* tier (least streaming) and record the numbers that forced it.
The decision is pure (no torch allocation), so tests feed mocked budgets and
assert the deciding math. ``tier=`` overrides ``auto``; real OOM at run time
triggers graceful demotion down :func:`demotion_ladder`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import StreamConfig
from .errors import TierSelectionError
from .hardware import HardwareInfo
from .logging_utils import get_logger
from .memory import MemoryEstimate

_log = get_logger("tiering")

TIER_NAMES = {0: "full", 1: "gpu_ram", 2: "ram", 3: "disk"}
_NAME_TO_TIER = {v: k for k, v in TIER_NAMES.items()}
_ALIASES = {"full": 0, "gpu_ram": 1, "ram": 2, "disk": 3}


def normalize_tier(tier: object) -> int | None:
    """Map ``"auto"``/int/name → tier int, or ``None`` for auto."""
    if tier is None or tier == "auto":
        return None
    if isinstance(tier, bool):  # guard: bool is an int subclass
        raise TierSelectionError(f"invalid tier {tier!r}")
    if isinstance(tier, int):
        if tier in TIER_NAMES:
            return tier
        raise TierSelectionError(f"tier must be 0..3 or a name; got {tier}")
    s = str(tier).lower()
    if s in _ALIASES:
        return _ALIASES[s]
    if s in _NAME_TO_TIER:
        return _NAME_TO_TIER[s]
    raise TierSelectionError(f"unknown tier {tier!r}; use auto|0|1|2|3|full|gpu_ram|ram|disk")


@dataclass(slots=True)
class TierDecision:
    """The chosen tier plus the budget breakdown that justifies it."""

    tier: int
    name: str
    backing: str  # "device" | "ram" | "disk"
    compute_device: str
    cache_layers: int | None
    reason: str
    numbers: dict[str, float] = field(default_factory=dict)
    honest_note: str | None = None
    forced: bool = False

    def summary(self) -> str:
        cl = "" if self.cache_layers is None else f", cache_layers={self.cache_layers}"
        return (
            f"Tier {self.tier} ({self.name}, backing={self.backing}, "
            f"device={self.compute_device}{cl})"
        )


def _cache_layers_on(budget: int, est: MemoryEstimate, overhead: int) -> int:
    """How many decoder layers fit on a device with ``budget`` usable bytes."""
    free = budget - est.resident_bytes - est.kv_bytes - est.activation_bytes - overhead
    if est.per_layer_bytes <= 0:
        return 0
    return free // est.per_layer_bytes


def select_tier(
    hw: HardwareInfo,
    est: MemoryEstimate,
    cfg: StreamConfig,
    *,
    device: str = "auto",
    tier_override: object = None,
) -> TierDecision:
    """Select a tier. ``tier_override`` (auto|0..3|name) forces a specific tier.

    Returns a :class:`TierDecision` whose ``numbers`` carry every term from §6 so
    ``.describe()`` and logs can show exactly why this tier and not another.
    """
    compute_device = hw.resolve_device(device)
    is_cuda = compute_device.startswith("cuda")
    is_unified = hw.unified_memory
    overhead = cfg.overhead_reserve_bytes(is_cuda)

    dev_avail = hw.device_available_bytes(compute_device)
    ram_avail = hw.available_ram_bytes
    usable_dev = int(dev_avail * cfg.headroom)
    usable_ram = int(ram_avail * cfg.headroom)
    disk_free = hw.disk_free_bytes

    nums: dict[str, float] = {
        "weights_bytes": est.weights_bytes,
        "per_layer_bytes": est.per_layer_bytes,
        "resident_bytes": est.resident_bytes,
        "kv_max_bytes": est.kv_bytes,
        "activation_bytes": est.activation_bytes,
        "overhead_reserve": overhead,
        "device_avail": dev_avail,
        "ram_avail": ram_avail,
        "usable_device": usable_dev,
        "usable_ram": usable_ram,
        "disk_free": disk_free,
        "headroom": cfg.headroom,
        "tier0_peak": est.tier0_peak_bytes(overhead),
    }

    forced = normalize_tier(tier_override)
    if forced is not None:
        decision = _build_tier(
            forced,
            hw,
            est,
            cfg,
            compute_device,
            is_cuda,
            is_unified,
            overhead,
            usable_dev,
            usable_ram,
            disk_free,
            nums,
            forced=True,
        )
        _log_decision(decision, est)
        return decision

    # --- auto: pick the least-streaming feasible tier ----------------------
    tier0_peak = est.tier0_peak_bytes(overhead)
    cache_dev = _cache_layers_on(usable_dev, est, overhead)
    cache_ram = _cache_layers_on(usable_ram, est, overhead)
    # The streaming "home" for Tiers 1/2 is host RAM, which must hold the decoder
    # *weights*. KV/activations live on the compute device (or, for unified/CPU,
    # the same pool) and are bounded separately by cache_layers + demotion.
    weights_fit_ram = est.weights_bytes <= usable_ram

    if tier0_peak <= usable_dev:
        chosen = 0
    elif weights_fit_ram and is_cuda and not is_unified and cache_dev >= cfg.min_cache_layers:
        chosen = 1
    elif weights_fit_ram:
        chosen = 2
    else:
        chosen = 3

    decision = _build_tier(
        chosen,
        hw,
        est,
        cfg,
        compute_device,
        is_cuda,
        is_unified,
        overhead,
        usable_dev,
        usable_ram,
        disk_free,
        nums,
        forced=False,
        cache_dev=cache_dev,
        cache_ram=cache_ram,
    )
    _log_decision(decision, est)
    return decision


def _build_tier(
    tier: int,
    hw: HardwareInfo,
    est: MemoryEstimate,
    cfg: StreamConfig,
    compute_device: str,
    is_cuda: bool,
    is_unified: bool,
    overhead: int,
    usable_dev: int,
    usable_ram: int,
    disk_free: int,
    nums: dict[str, float],
    *,
    forced: bool,
    cache_dev: int | None = None,
    cache_ram: int | None = None,
) -> TierDecision:
    if cache_dev is None:
        cache_dev = _cache_layers_on(usable_dev, est, overhead)
    if cache_ram is None:
        cache_ram = _cache_layers_on(usable_ram, est, overhead)

    gb = 1e9
    fit = est.tier0_peak_bytes(overhead)

    if tier == 0:
        feasible = fit <= usable_dev
        note: str | None = (
            f"Model fits in {compute_device} memory; streamllm adds no value here "
            "— vLLM/TGI/llama.cpp will be faster. Loading normally."
            if is_cuda or compute_device == "mps"
            else "Loaded fully in host RAM and running on CPU; a dedicated CPU "
            "engine (llama.cpp) would be faster. Loading normally."
        )
        reason = (
            f"weights+KV+act+overhead={fit / gb:.2f}GB "
            f"{'<=' if feasible else '>'} usable_device={usable_dev / gb:.2f}GB"
        )
        if forced and not feasible:
            reason = "FORCED Tier 0 despite " + reason + " (may OOM)"
        return TierDecision(0, "full", "device", compute_device, None, reason, nums, note, forced)

    if tier == 1:
        # GPU resident set + stream the rest from pinned host RAM.
        cl = max(int(cache_dev), cfg.min_cache_layers)
        weights_fit_ram = est.weights_bytes <= usable_ram
        feasible = (
            is_cuda and not is_unified and cache_dev >= cfg.min_cache_layers and weights_fit_ram
        )
        reason = (
            f"cache_layers=floor((usable_vram {usable_dev / gb:.2f}GB - resident "
            f"{est.resident_bytes / gb:.2f} - KV {est.kv_bytes / gb:.2f} - act "
            f"{est.activation_bytes / gb:.2f} - overhead {overhead / gb:.2f})/per_layer "
            f"{est.per_layer_bytes / gb:.3f})={cache_dev}; weights_fit_ram={weights_fit_ram}"
        )
        if not feasible and not forced:
            reason += " -> Tier 1 infeasible"
        nums = {**nums, "cache_layers": cache_dev}
        return TierDecision(1, "gpu_ram", "ram", compute_device, cl, reason, nums, None, forced)

    if tier == 2:
        # All decoder weights in RAM; stream into compute. Unified = residency only.
        cl = max(int(cache_dev if is_cuda else cache_ram), 1)
        note = None
        if is_unified:
            note = "Unified memory: streaming is residency management, not a real copy."
        weights_fit_ram = est.weights_bytes <= usable_ram
        reason = (
            f"weights {est.weights_bytes / gb:.2f}GB {'<=' if weights_fit_ram else '>'} "
            f"usable_ram {usable_ram / gb:.2f}GB; stream layer-by-layer "
            f"(resident cache_layers={cl}, unified={is_unified})"
        )
        if forced and not weights_fit_ram:
            reason = "FORCED Tier 2 despite weights>usable_ram (may OOM)"
        nums = {**nums, "cache_layers": cl}
        return TierDecision(2, "ram", "ram", compute_device, cl, reason, nums, note, forced)

    # tier == 3: disk-backed streaming
    cl = max(int(cache_dev if is_cuda else cache_ram), 1)
    note = (
        "Tier 3 is I/O-bound by physics: every layer is read from disk per token. "
        "Quantized shards, batch amortization and prefetch reduce the wall; none remove it."
    )
    enough_disk = disk_free >= est.weights_bytes
    reason = (
        f"weights {est.weights_bytes / gb:.2f}GB > usable_ram {usable_ram / gb:.2f}GB; "
        f"stream from mmap'd shards (disk_free {disk_free / gb:.2f}GB, "
        f"{'sufficient' if enough_disk else 'INSUFFICIENT for full-precision shards'})"
    )
    nums = {**nums, "cache_layers": cl}
    return TierDecision(3, "disk", "disk", compute_device, cl, reason, nums, note, forced)


def demotion_ladder(current: int) -> list[int]:
    """Tiers to try, in order, when ``current`` OOMs at run time (prompt §10)."""
    return [t for t in (current + 1, current + 2, current + 3) if t <= 3]


def _log_decision(decision: TierDecision, est: MemoryEstimate) -> None:
    _log.info("%s -> %s", decision.summary(), decision.reason)
    if decision.honest_note:
        _log.info("note: %s", decision.honest_note)
    if est.is_moe:
        _log.warning(
            "model is MoE (%d experts/layer); expert-level streaming is NOT implemented",
            est.num_experts,
        )
