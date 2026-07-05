# CLAUDE.md — streamllm

Speed-first, auto-tiering LLM inference. The headline feature is the **policy that
decides how little to stream**, not streaming itself. Read `SPEC.md` for the
crystallized spec and `docs/architecture.md` for how it fits together.

## Layout

```
src/streamllm/
  config.py      StreamConfig (tunables + env overrides)
  errors.py      exception hierarchy (every message is actionable)
  hardware.py    detect_hardware -> HardwareInfo (cuda/mps/ram/disk)
  memory.py      estimate_memory -> MemoryEstimate (the budget model, §6)
  graph.py       discover_graph (generic decoder-stack discovery, no hardcoded names)
  tiering.py     select_tier -> TierDecision + demotion_ladder
  cache.py       LayerCache (thread-safe LRU + refcount)
  prefetch.py    Prefetcher (background, dedicated CUDA stream)
  runner.py      StreamingRunner + LayerHandle (weight streaming via forward hooks)
  generation.py  sampling / stopping / left-pad batch / streamer
  quant.py       int8/int4 (pure torch) + nf4 (bitsandbytes optional)
  shard.py       disk shard format + build/resume/verify + mmap load
  metrics.py     RunMetrics
  benchmark.py   benchmark_model (median+p90, TTFT, verdict, batch sweep)
  model.py       StreamModel (from_pretrained/from_model/from_shards, generate, describe)
  cli.py         `streamllm` entry: run | shard | bench | describe
bench/benchmark.py   thin CLI wrapper
web/                 landing + live tier calculator + FastAPI playground (DESIGN.md)
tests/               CPU-only, no downloads, tiny random configs + mocked budgets
```

## Working on this repo

- **Run tests:** `PYTHONPATH=src python3 -m pytest -q` (CPU-only, no network).
- **Lint/type:** `ruff check src tests bench && ruff format src tests bench && mypy src`.
  Both must be clean before declaring done.
- **The correctness oracle is `tests/test_verify.py`** — streamed logits must
  match a plain full load. If you touch the runner, the cache, the prefetcher, or
  the generation loop, this is the test that catches silent wrongness. Keep it green.
- **The tiering math is pure** — `select_tier` does no torch allocation, so
  `tests/test_tiering.py` feeds mocked budgets. Add a test there for any policy change.
- **transformers is 5.x here.** Decoder layers receive a precomputed
  `position_embeddings`; the cache is `DynamicCache`. Do NOT reimplement the model
  forward; stream weights via hooks so the model's own mask/RoPE/KV math runs.
- **This box is a DGX Spark (GB10, unified memory).** `torch.cuda.mem_get_info`
  can OOM at detection; the code degrades to CPU. Tier 1/3 are tested via mocked
  budgets + tiny models, not real big-model runs.

## Conventions

- Python 3.11+, full type hints, `from __future__ import annotations`.
- Log under the `streamllm` logger; never `print` except in `cli.py`.
- Every automatic decision is logged with the numbers that forced it, and is
  overridable (`tier=`, `cache_layers=`, `headroom=`, graph overrides).
- Honest by construction: Tier 0 says "use a real engine"; Tier 3 says "I/O-bound
  by physics". Do not soften these.
- No em-dashes in the web copy (`DESIGN.md` is the frontend design system).

## Playground

`pip install "streamllm[web]"` then `STREAMLLM_MODEL=<id> python web/server.py`
(default port 6909). Serves the landing page, the client-side tier calculator
(a JS port of the §6 estimator that matches `streamllm describe`), and the
SSE-streaming chat playground.
