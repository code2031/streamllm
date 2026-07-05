"""Disk shard format + build/resume + mmap load (prompt §9) — the Tier 3 lever.

Layout under ``shard_path``:

* ``layer_NNNN.safetensors`` — one file per decoder layer (keys = param names).
* ``resident.safetensors``   — embeddings + final norm + LM head (the resident set).
* ``config.json``            — the HF config, so reload needs no original checkpoint.
* ``streamllm_manifest.json``— source id/revision, dtype, quant scheme, per-shard
  byte sizes + sha256, format version. Resumability + integrity hang off this.

``mmap`` load uses ``safetensors.safe_open`` so the OS page cache does the right
thing — we never read a whole shard into Python and then copy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .config import StreamConfig
from .errors import ShardError
from .logging_utils import get_logger

if TYPE_CHECKING:
    from .graph import ModelGraph
    from .hardware import HardwareInfo
    from .runner import StreamingRunner
    from .tiering import TierDecision

_log = get_logger("shard")

FORMAT_VERSION = 1
MANIFEST_NAME = "streamllm_manifest.json"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


@dataclass(slots=True)
class ShardResult:
    """Outcome of a (possibly resumed) shard build."""

    manifest_path: Path
    manifest: dict[str, Any]
    built: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return (
            sum(layer["bytes"] for layer in self.manifest["layers"])
            + self.manifest["resident"]["bytes"]
        )


def shard_model(
    model: torch.nn.Module,
    config: Any,
    *,
    out_path: str | Path,
    dtype: torch.dtype | None = None,
    quantize: str | None = None,
    delete_source: bool = False,
    source_model: str | None = None,
    revision: str | None = None,
    stream_config: StreamConfig | None = None,
    graph: ModelGraph | None = None,
    progress: bool = False,
) -> ShardResult:
    """Decompose ``model`` into per-layer safetensors shards (prompt §9).

    Refuses with a clear message if free disk is insufficient; is resumable (skips
    shards whose hash already matches the manifest); validates each shard by
    re-reading + hashing before marking it done. ``delete_source`` is honored only
    after every shard is written and verified.
    """
    from .graph import discover_graph

    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    g = graph or discover_graph(model, config)
    quant_bytes = {None: None, "int8": 1.0, "int4": 0.5}
    if quantize not in quant_bytes:
        raise ShardError(f"quantize must be one of {list(quant_bytes)}; got {quantize!r}")

    # 1) Disk free check BEFORE writing anything.
    est_bytes = _estimate_disk_bytes(model, quantize)
    free = shutil.disk_usage(out).free
    if est_bytes > free * 0.98:
        raise ShardError(
            f"insufficient disk to shard: need ~{est_bytes / 1e9:.2f}GB, "
            f"have {free / 1e9:.2f}GB free at {out}. Free space or pick another --out."
        )

    existing = _load_manifest_if_present(out)
    existing_layers = {layer["file"]: layer for layer in (existing or {}).get("layers", [])}

    from .quant import maybe_quantize_state

    layers_meta: list[dict[str, Any]] = []
    built: list[str] = []
    skipped: list[str] = []

    iterator = list(enumerate(g.layers))
    if progress:
        try:
            from tqdm import tqdm

            iterator = list(tqdm(iterator, desc="sharding layers", unit="layer"))
        except ImportError:  # pragma: no cover
            pass

    for i, layer in iterator:
        fname = f"layer_{i:04d}.safetensors"
        fpath = out / fname
        param_names = [n for n, _ in layer.named_parameters()]
        # Resume: skip if file present and its hash matches a prior manifest entry.
        if (
            fpath.exists()
            and fname in existing_layers
            and _sha256_file(fpath) == existing_layers[fname]["sha256"]
        ):
            layers_meta.append(existing_layers[fname])
            skipped.append(fname)
            continue
        tensors = {n: _materialize_for_save(p, dtype) for n, p in layer.named_parameters()}
        tensors, quant_meta = maybe_quantize_state(tensors, quantize)
        _save_and_verify(fpath, tensors)
        layers_meta.append(
            {
                "file": fname,
                "bytes": fpath.stat().st_size,
                "sha256": _sha256_file(fpath),
                "param_names": param_names,
                "quant": quant_meta,
            }
        )
        built.append(fname)

    # 2) Resident set (embed/norm/head), tie-aware.
    resident_tensors, tied = _collect_resident(g, dtype)
    rfile = out / "resident.safetensors"
    _save_and_verify(rfile, resident_tensors)
    resident_meta = {
        "file": "resident.safetensors",
        "bytes": rfile.stat().st_size,
        "sha256": _sha256_file(rfile),
        "tied_embeddings": tied,
        "keys": list(resident_tensors.keys()),
    }

    # 3) Persist config + manifest.
    try:
        config.to_json_file(str(out / "config.json"))
    except Exception:  # pragma: no cover - duck-typed configs in tests
        (out / "config.json").write_text(json.dumps(_config_to_dict(config)))

    manifest = {
        "format_version": FORMAT_VERSION,
        "source_model": source_model or "<in-memory>",
        "revision": revision,
        "dtype": str(dtype or _model_dtype(model)).replace("torch.", ""),
        "quantization": {"scheme": quantize},
        "num_layers": len(g.layers),
        "layers_path": g.layers_path,
        "layers": layers_meta,
        "resident": resident_meta,
    }
    manifest_path = out / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _log.info(
        "sharded %d layers to %s (built=%d skipped=%d)",
        len(g.layers),
        out,
        len(built),
        len(skipped),
    )

    if delete_source:
        _delete_source(source_model)

    return ShardResult(manifest_path=manifest_path, manifest=manifest, built=built, skipped=skipped)


def load_manifest(shard_path: str | Path) -> dict[str, Any]:
    """Load + validate the shard manifest, raising on version/integrity issues."""
    out = Path(shard_path)
    mpath = out / MANIFEST_NAME
    if not mpath.exists():
        raise ShardError(f"no {MANIFEST_NAME} at {out}; was the model sharded there?")
    manifest = json.loads(mpath.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ShardError(
            f"shard format_version {manifest.get('format_version')} != {FORMAT_VERSION}; re-shard"
        )
    return manifest


def verify_shards(shard_path: str | Path) -> bool:
    """Re-hash every shard file against the manifest. Returns True if all match."""
    out = Path(shard_path)
    manifest = load_manifest(out)
    for layer in manifest["layers"] + [manifest["resident"]]:
        fpath = out / layer["file"]
        if not fpath.exists():
            raise ShardError(f"missing shard {layer['file']}")
        if _sha256_file(fpath) != layer["sha256"]:
            raise ShardError(f"hash mismatch for {layer['file']} (corrupt shard)")
    return True


def _make_layer_fetch(
    layer_file: Path, param_names: list[str], quant_meta: Any, io_retry: int
) -> Any:
    """Return a closure that mmaps ``layer_file`` and returns tensors in order."""
    from .quant import dequantize_tensor

    def fetch() -> list[torch.Tensor]:
        from safetensors import safe_open

        last_err: Exception | None = None
        for attempt in range(io_retry + 1):
            try:
                out: list[torch.Tensor] = []
                with safe_open(str(layer_file), framework="pt", device="cpu") as f:
                    for name in param_names:
                        t = f.get_tensor(name)
                        if quant_meta:
                            t = dequantize_tensor(t, f, name, quant_meta)
                        out.append(t)
                return out
            except Exception as exc:  # pragma: no cover - IO-specific
                last_err = exc
                _log.warning("shard read %s failed (attempt %d): %s", layer_file, attempt, exc)
        raise ShardError(
            f"failed to read shard {layer_file} after {io_retry + 1} attempts: {last_err}"
        )

    return fetch


def load_sharded_runtime(
    shard_path: str | Path,
    config: Any,
    decision: TierDecision,
    cfg: StreamConfig,
    hw: HardwareInfo,
    *,
    device: str,
    estimate: Any,
) -> tuple[torch.nn.Module, StreamingRunner, ModelGraph]:
    """Build a skeleton + disk-backed streaming runner from shards (prompt §9).

    Caveat: we instantiate the model structure with ``from_config`` (real params +
    correctly computed RoPE buffers) and immediately free the decoder weights to
    placeholders before streaming from disk. This briefly touches full-model RAM
    during reload; quantized shards reduce it. A fully meta-skeleton reload that
    recomputes RoPE buffers without allocating any layer is future work.
    """
    from transformers import AutoModelForCausalLM

    from .graph import discover_graph
    from .runner import StreamingRunner

    out = Path(shard_path)
    manifest = load_manifest(out)
    model = AutoModelForCausalLM.from_config(config)
    graph = discover_graph(model, config)

    # Disk-backed fetchers: each materialize reads its layer shard via mmap.
    fetchers = []
    for i, _layer in enumerate(graph.layers):
        meta = manifest["layers"][i]
        fetchers.append(
            _make_layer_fetch(
                out / meta["file"], meta["param_names"], meta.get("quant"), cfg.io_retry_count
            )
        )

    _load_resident(out, manifest, graph, device)

    # Tier 3 must stream/evict even on unified memory (it didn't fit RAM), so we
    # force a non-residency transfer mode by passing unified=False.
    runner = StreamingRunner(
        model, graph, decision, cfg, device, estimate, unified=False, layer_fetchers=fetchers
    )
    runner.install()
    return model, runner, graph


def build_disk_model(
    model_id_or_path: str,
    config: Any,
    decision: TierDecision,
    cfg: StreamConfig,
    hw: HardwareInfo,
    *,
    shard_path: str | None,
    dtype: torch.dtype,
    quantize: str | None,
    trust_remote_code: bool,
) -> tuple[torch.nn.Module, StreamingRunner, ModelGraph]:
    """Ensure shards exist for ``model_id_or_path`` then load a disk runtime."""
    out = Path(shard_path) if shard_path else (cfg.cache_dir / _safe_name(model_id_or_path))
    if not (out / MANIFEST_NAME).exists():
        _log.warning(
            "no shards at %s; building them (this loads the model once and needs "
            "enough RAM/disk for the shard step)",
            out,
        )
        from transformers import AutoModelForCausalLM

        src = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        shard_model(
            src,
            config,
            out_path=out,
            dtype=dtype,
            quantize=quantize,
            source_model=model_id_or_path,
            progress=True,
        )
        del src
    from .model import _build_estimate  # local import to avoid cycle

    est = _build_estimate(
        config, _peek_graph(config), cfg, dtype, quantize, cfg.max_context_default
    )
    model, runner, graph = load_sharded_runtime(
        out, config, decision, cfg, hw, device=decision.compute_device, estimate=est
    )
    return model, runner, graph


# --------------------------------------------------------------------- helpers


def _peek_graph(config: Any) -> ModelGraph:
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    from .graph import discover_graph

    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(config)
    return discover_graph(m, config)


def _materialize_for_save(param: torch.nn.Parameter, dtype: torch.dtype | None) -> torch.Tensor:
    t = param.detach()
    if t.is_meta:
        raise ShardError("cannot shard a meta-device parameter; load real weights before sharding")
    t = t.to("cpu")
    if dtype is not None and t.is_floating_point():
        t = t.to(dtype)
    return t.contiguous()


def _collect_resident(graph: ModelGraph, dtype: torch.dtype | None) -> tuple[dict, bool]:
    tensors: dict[str, torch.Tensor] = {}
    for n, p in graph.embed_tokens.named_parameters():
        tensors[f"embed.{n}"] = _materialize_for_save(p, dtype)
    for n, p in graph.final_norm.named_parameters():
        tensors[f"norm.{n}"] = _materialize_for_save(p, dtype)
    embed_ids = {id(p) for p in graph.embed_tokens.parameters()}
    tied = all(id(p) in embed_ids for p in graph.lm_head.parameters())
    if not tied:
        for n, p in graph.lm_head.named_parameters():
            tensors[f"lm_head.{n}"] = _materialize_for_save(p, dtype)
    return tensors, tied


def _load_resident(out: Path, manifest: dict, graph: ModelGraph, device: str) -> None:
    from safetensors import safe_open

    rfile = out / manifest["resident"]["file"]
    with safe_open(str(rfile), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for n, p in graph.embed_tokens.named_parameters():
            p.data = f.get_tensor(f"embed.{n}").to(device)
        for n, p in graph.final_norm.named_parameters():
            p.data = f.get_tensor(f"norm.{n}").to(device)
        if manifest["resident"].get("tied_embeddings"):
            graph.lm_head.weight.data = graph.embed_tokens.weight.data
        else:
            for n, p in graph.lm_head.named_parameters():
                key = f"lm_head.{n}"
                if key in keys:
                    p.data = f.get_tensor(key).to(device)
    if graph.rotary_emb is not None:
        graph.rotary_emb.to(device)


def _save_and_verify(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
    # Validate by re-reading before committing the final name (atomic rename).
    from safetensors import safe_open

    with safe_open(str(tmp), framework="pt", device="cpu") as f:
        if set(f.keys()) != set(tensors.keys()):
            raise ShardError(f"shard verify failed for {path}: key mismatch on re-read")
    tmp.replace(path)


def _estimate_disk_bytes(model: torch.nn.Module, quantize: str | None) -> int:
    factor = {None: 1.0, "int8": 0.5, "int4": 0.28}.get(quantize, 1.0)
    total = 0
    for p in model.parameters():
        elt = 1 if p.is_meta else p.element_size()
        total += p.numel() * elt
    return int(total * factor) + (16 << 20)  # + manifest/overhead slack


def _load_manifest_if_present(out: Path) -> dict | None:
    mpath = out / MANIFEST_NAME
    if mpath.exists():
        try:
            return json.loads(mpath.read_text())
        except json.JSONDecodeError:  # pragma: no cover
            return None
    return None


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    for p in model.parameters():
        return p.dtype
    return torch.float32


def _config_to_dict(config: Any) -> dict:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return {k: v for k, v in vars(config).items() if not k.startswith("_")}


def _safe_name(model_id: str) -> str:
    return "shards--" + model_id.replace("/", "--").replace("\\", "--")


def _delete_source(source_model: str | None) -> None:
    # We only delete an HF *cache* entry, never an arbitrary path the user passed.
    if not source_model or "/" not in source_model or Path(source_model).exists():
        _log.warning("delete_source skipped: %r is not a deletable HF cache id", source_model)
        return
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == source_model:
                shutil.rmtree(repo.repo_path, ignore_errors=True)
                _log.info("deleted source HF cache for %s", source_model)
    except Exception as exc:  # pragma: no cover
        _log.warning("delete_source failed for %s: %s", source_model, exc)
