# Changelog

All notable changes to streamllm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `streamllm describe` / `estimate_only` now build a cheap meta-device skeleton
  (still zero memory, no weights) to get **exact** per-layer/resident param counts,
  falling back to the analytic SwiGLU formula only if the skeleton cannot be built.
  This makes the memory estimate accurate for any architecture (e.g. GPT-2's
  non-gated MLP), not just Llama/Qwen. The `budget.source` field reports `measured`
  vs `analytic`.

## [0.1.0] - 2026-06-24

First release. Speed-first, auto-tiering inference for large language models.

### Packaging & CI
- GitHub Actions CI running ruff, mypy, and the pytest suite on CPU (no GPU, no
  downloads) across Python 3.11 and 3.12.
- `py.typed` marker so downstream users get streamllm's type information (PEP 561).
- `streamllm --version` flag.
- `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, `CHANGELOG.md`, `Makefile`.
- pytest `pythonpath = ["src", "."]` so the suite runs under a bare `pytest` in CI
  (not only `python -m pytest`); the web extra is installed in CI so the playground
  TestClient tests run there too.

### Security
- Playground server (`web/server.py`) is hardened: `max_new_tokens` and prompt
  length are clamped, generation is serialized (HTTP 429 when busy) so load cannot
  pile up against the single shared model, generation errors return a generic
  message (details logged server-side, never leaked to the client), and
  `/api/describe` no longer exposes an absolute local model path.

### Added
- Auto-tiering policy (`tiering.py`) selecting the least-streaming tier (0 full /
  1 gpu_ram / 2 ram / 3 disk) from a peak-memory budget, logging the deciding numbers.
- Peak-memory budget model (`memory.py`) with GQA-correct KV, tied-embedding and
  sliding-window handling, MoE detection, analytic + measured sources.
- Generic module-graph discovery (`graph.py`) with no hardcoded layer names, plus
  overrides for unrecognized architectures.
- Hook-based streaming runner (`runner.py`) with a thread-safe LRU cache
  (`cache.py`), a dedicated-CUDA-stream prefetcher (`prefetch.py`), and three
  transfer modes (cuda / copy / residency).
- Tier-agnostic generation (`generation.py`, `model.py`): sampling, stop strings,
  left-padded batching, seeded determinism, token streaming.
- Disk shard format (`shard.py`): per-layer safetensors, hashed manifest,
  resumable build, mmap load, `from_shards` reload.
- Weight-only quantization (`quant.py`): pure-torch int8/int4, optional nf4 via
  bitsandbytes.
- Graceful tier demotion on real OOM, `RunMetrics`, benchmark with I/O-vs-compute
  verdict and Tier-3 batch sweep.
- `streamllm` CLI: `run` / `shard` / `bench` / `describe`.
- Web frontend (`web/`): static landing page, client-side tier calculator that
  matches `streamllm describe`, and a FastAPI SSE playground.
- 91 CPU-only tests (tiny random configs + mocked budgets); ruff + mypy clean.
  Coverage includes module-graph discovery on Llama/Qwen/GPT-2, sampling filters,
  sliding-window, LRU-cache concurrency, playground TestClient, and shard reload.
