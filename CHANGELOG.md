# Changelog

All notable changes to streamllm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI running ruff, mypy, and the pytest suite on CPU (no GPU, no
  downloads) across Python 3.11 and 3.12.
- `py.typed` marker so downstream users get streamllm's type information (PEP 561).
- `streamllm --version` flag.
- `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, `CHANGELOG.md`.
- Expanded test coverage (74 -> 91): module-graph discovery on a third
  architecture (GPT-2), top-k/top-p/min-p filter semantics, hypothesis property
  tests for KV linear scaling, sliding-window streamed-vs-full forward, LRU-cache
  concurrency stress, playground TestClient smoke, and int4 shard reload.

### Security
- Playground server (`web/server.py`) hardened: `max_new_tokens` and prompt length
  are clamped, generation is serialized (HTTP 429 when busy) so load cannot pile
  up against the single shared model, generation errors return a generic message
  (details logged server-side, never leaked to the client), and `/api/describe`
  no longer exposes an absolute local model path.

## [0.1.0] - 2026-06-24

First release. Speed-first, auto-tiering inference for large language models.

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
- 74 CPU-only tests; ruff + mypy clean.
