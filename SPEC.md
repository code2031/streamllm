# streamllm — Crystallized Build Spec

> Authored via `/spec`, grounded in the actual build environment. The source of
> truth for *requirements* is the build prompt (21 sections). This file pins the
> **decisions** that the build prompt left open and records what was verified on
> this machine. `DESIGN.md` is a separate artifact: the Vercel-inspired design
> system for the web frontend (landing page + playground), not this spec.

## 1. Mission (unchanged from prompt §1)

Run a model as fast as the hardware allows. When it fits, **don't stream** — load
normally and tell the user a dedicated engine (vLLM/TGI/llama.cpp) is faster. When
it doesn't fit, fall back through tiers that stream as *little* as possible. The
headline feature is the **policy that decides how little to stream**. Be honest by
construction: the disk tier is I/O-bound by physics; logs and reports say so.

## 2. Verified environment (this box, 2026-06)

| Thing | Value | Consequence |
|---|---|---|
| Python | 3.12.3 | target `>=3.11`, fine |
| torch | 2.11.0+cu130, CUDA **available** | Tier 0/2 runnable for real |
| Device 0 | NVIDIA **GB10** (DGX Spark), unified mem | unified-memory path matters; `mem_get_info` works |
| RAM | 131 GB total, ~13 GB available at probe | detection must use **available**, not total |
| transformers | **5.3.0** (major 5.x) | layer fwd takes precomputed `position_embeddings`; cache is `DynamicCache`; mask built internally |
| accelerate | 1.13.0 | `init_empty_weights` present |
| safetensors | 0.7.0 | mmap/lazy load available |
| psutil / numpy / tqdm | 5.9 / 2.4 / 4.67 | present |
| bitsandbytes | **absent** | `[quant]` optional; clear error if requested without it |
| hypothesis | **absent** | property tests `importorskip`; parametrized fallback always runs |
| ruff / mypy | 0.15 / 2.1 | lint+type gates runnable |

CI assumption holds: the full suite must pass on **CPU, no downloads**, using
tiny random-weight configs and **mocked** hardware budgets for Tiers 1/3.

## 3. Key open-decisions, now pinned

1. **Streaming mechanism = forward hooks, not a reimplemented forward.**
   transformers 5.x builds masks and RoPE internally and passes precomputed
   `position_embeddings` into each decoder layer. Reimplementing that per
   architecture is the #1 silent-wrongness risk. Instead we keep the model's own
   `forward` intact and stream *where each decoder layer's weights live* via
   `forward_pre_hook` (materialize on compute device) + `forward_hook` (release to
   LRU). This makes Tier 1/2/3 architecture-agnostic and lets
   `verify_against_full_load` actually pass. Downside: we depend on the model
   exposing decoder layers as discrete `nn.Module`s (true for all HF causal LMs).
2. **Generation loop is ours, not HF `.generate()`.** We need streaming callbacks,
   per-tier metrics, left-padded batched decode, and an externally-held resident
   KV cache (`DynamicCache`). We drive `model(...)` step-by-step.
3. **Memory estimate has two sources.** `describe`/estimate-only uses an **analytic**
   per-layer/resident param count from `config.json` (no weights, no model build).
   At load time, when a meta skeleton exists, we refine with **measured** param
   counts. Both feed the same `MemoryEstimate`. Documented divergence: analytic
   assumes SwiGLU-style llama/qwen layers (q/k/v/o + gate/up/down + 2 norms).
4. **"CPU device" exercises the streaming path in CI.** With no CUDA, the compute
   device is CPU and "evicted" weights live in a home dict / on `meta`;
   materialize = real copy onto CPU. Same LRU/refcount/prefetch code path, no real
   H2D transfer. Prefetch overlap only yields wall-clock wins on CUDA (documented).
5. **Spec artifact = this file** (local), not a GitHub issue — streamllm has no
   repo of its own and lives inside `~/`. Frontend uses `DESIGN.md`.

## 4. Memory budget formulas (binding, from §6)

```
per_layer_bytes   = decoder_layer_param_count × dtype_bytes(after quant)
resident_bytes    = embed + final_norm + lm_head      # share tensor if tie_word_embeddings
kv_bytes(ctx)     = 2 × n_layers × num_key_value_heads × head_dim
                      × min(ctx, sliding_window or ctx) × batch × kv_dtype_bytes
activation_peak   = batch × prompt_len × hidden × dtype_bytes × activation_factor   # factor ~2–3
overhead_reserve  = fixed floor (CUDA ~1.0 GB, else ~0.5 GB)
usable(avail)     = avail × headroom            # headroom default 0.9
```

Use `num_key_value_heads` (GQA/MQA), **never** `num_attention_heads`, in KV math.
Tier-1 resident layer solve:
```
cache_layers = floor((usable_VRAM − resident − kv_max − activation − overhead) / per_layer_bytes)
cache_layers = max(cache_layers, 2)          # ≥2 for double-buffered prefetch
```
`cache_layers < 2` ⇒ Tier 1 infeasible ⇒ fall to Tier 2/3.

## 5. Tier decision (from §5)

- **Tier 0** weights+kv_max+act+overhead ≤ usable VRAM → full load. Emit the
  "use vLLM/TGI/llama.cpp" honesty line. Still run.
- **Tier 1** doesn't fit VRAM, fits usable RAM, `cache_layers ≥ 2` on GPU → resident
  set on GPU + stream rest from **pinned host RAM** with LRU + async prefetch.
- **Tier 2** fits usable RAM, little/no GPU or unified → all decoder weights in RAM,
  stream into compute. Unified mem ⇒ residency only, skip redundant copies.
- **Tier 3** doesn't fit RAM → stream from mmap'd on-disk shards. I/O-bound; log the
  effective read GB/s ceiling. Levers: quantized shards, batch amortization, prefetch.

Every decision logs the deciding numbers and is overridable (`tier=`/`backing=`).
Real OOM at materialization ⇒ graceful one-tier **demotion** with a loud warning.

## 6. Module layout (`src/streamllm/`)

```
config.py     StreamConfig dataclass (+ env overrides), validation
errors.py     exception hierarchy
logging_utils log under `streamllm` logger; STREAMLLM_LOG_LEVEL
hardware.py   HardwareInfo detection (cuda mem_get_info, mps, psutil, disk)
memory.py     MemoryEstimate + analytic/measured estimator (the failure point)
graph.py      ModelGraph generic discovery (no hardcoded names) + overrides
tiering.py    TierDecision policy + demotion ladder
cache.py      thread-safe LRU layer cache (refcount, no evict-in-use)
prefetch.py   background stager: dedicated CUDA stream, pinned buffers, events
runner.py     StreamingRunner: hook install, resident/stream loop, KV held
generation.py sampling/stopping/stop-strings/left-pad batch/streamer/seed
quant.py      int8/int4 (bitsandbytes), shard-time + load-time, optional dep
shard.py      shard format + manifest + build/resume/verify + mmap load
metrics.py    RunMetrics (tokens, TTFT, per-layer load vs compute, hit rate, peak)
model.py      StreamModel.from_pretrained / generate / forward / describe / shard
cli.py        run | shard | bench | describe  (console entry `streamllm`)
```

`bench/benchmark.py` (median+p90, TTFT, I/O-vs-compute verdict, Tier-3 batch sweep,
JSON+CSV). `web/` holds the frontend (see §8).

## 7. Test matrix (from §15, all CPU/no-download)

1 tier selection (mocked budgets, all 4 tiers + deciding math) · 2 estimator
property tests (KV linear in ctx/batch/layers; GQA smaller; tied not double-counted;
sliding-window cap) · 3 **long-context OOM stress** (fits-at-load, KV growth would
exceed → picks lower tier or demotes, never OOM-crashes) · 4 module-graph discovery
(llama-like + qwen-like + unrecognized→override) · 5 LRU eviction (cap, no in-use
evict, hit/miss) · 6 **verify_against_full_load** prefill **and** multi-step decode
(logits match, documented tol) · 7 batched left-pad decode == per-seq · 8 sampling
determinism (seed; greedy deterministic) · 9 shard round-trip + resumability ·
10 prefetch-overlap smoke (stages ahead; event gates use).

## 8. Frontend deliverables (design system = `DESIGN.md`)

All three, styled with the Vercel-inspired tokens in `DESIGN.md` (ink/near-white,
Geist + Geist Mono, hero mesh gradient, stacked shadows, 100px pill CTAs, mono
eyebrows):

- **Landing page** (`web/index.html`, static): honest tier table, hardware-fit
  matrix (7B/13B/70B/405B × fp16/int8/int4), quickstart, limitations-first copy.
- **Live tier calculator** (client-side JS port of §4/§5 estimator): pick model
  preset / params / dtype / quant / VRAM / RAM → shows tier + full memory
  breakdown bar (weights / KV-at-max-ctx / activation / overhead) + the honesty
  verdict. No backend.
- **Web playground** (`web/playground/`, FastAPI `web/server.py`): thin server
  wrapping `StreamModel.generate` with SSE token streaming on the GB10, designed
  chat UI + a live `.describe()` panel. Server-bound; not part of library v1
  (kept in `web/`, not `src/`), respecting the no-server-in-library non-goal.

## 9. Definition of done (from §21)

`describe` without weights ✓ · all 4 tiers reached+logged under mocked budgets ✓ ·
Llama/Qwen generates with CLI token streaming ✓ · verify_against_full_load prefill
+ decode ✓ · long-context OOM never crashes ✓ · shards round-trip + resumable +
quantized smaller ✓ · bench tokens/s + TTFT + verdict + batch sweep + JSON/CSV ✓ ·
module-graph on 2 arches no hardcoded names + override ✓ · suite green on CPU,
ruff+mypy clean, THIRD_PARTY.md ✓ · landing + calculator + playground shipped ✓.
