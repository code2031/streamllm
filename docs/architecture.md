# Architecture

This document covers the runner, the tiering policy, prefetch/eviction, the shard
format, and the memory model with the actual formulas. The source of binding
requirements is the build prompt; `SPEC.md` records the decisions that were left
open.

```
from_pretrained / from_model
        |
        v
  detect_hardware ──► HardwareInfo (cuda mem_get_info, mps, psutil RAM, disk)
        |
        v
  build meta skeleton ──► discover_graph (decoder stack / embed / norm / head)
        |
        v
  estimate_memory ──► MemoryEstimate (per-layer, resident, KV@maxctx, activation)
        |
        v
  select_tier ──► TierDecision (the number that forced the choice)
        |
        +── tier 0 ──► model.to(device), no streaming
        |
        +── tier 1/2/3 ──► StreamingRunner.install()
                              |
                              +── LayerHandle per decoder layer (weight movement)
                              +── LayerCache (LRU + refcount)
                              +── Prefetcher (background, dedicated CUDA stream)
        |
        v
  generate ──► tier-agnostic decode loop over model(...) with DynamicCache
```

## The memory model (`memory.py`, prompt §6)

Estimates **peak** memory, not load-time, accounting for KV growth over the full
generation. Two sources: **measured** (exact param counts from a meta-device
skeleton, still zero memory) and **analytic** (the SwiGLU formula from config
fields). Both `describe`/`estimate_only` and load-time try the measured skeleton
first and fall back to analytic only if the skeleton cannot be built, so the
estimate is accurate for any architecture (GPT-2, MoE), not just Llama/Qwen. The
`budget.source` field reports which was used.

```
per_layer_bytes  = decoder_layer_param_count * weight_bytes_per_param
resident_bytes   = (embed + final_norm + lm_head) * weight_bytes_per_param   # tie-aware
weights_bytes    = per_layer_bytes * num_hidden_layers + resident_bytes

kv_bytes(ctx)    = 2 * num_hidden_layers * num_key_value_heads * head_dim
                     * min(ctx, sliding_window or ctx) * batch * kv_dtype_bytes
activation_bytes = batch * prompt_len * hidden_size * dtype_bytes * activation_factor
overhead_reserve = 1.0 GB on CUDA, 0.5 GB otherwise
usable(avail)    = avail * headroom        # headroom default 0.9
```

The analytic decoder-layer count assumes a gated-MLP (SwiGLU) llama/qwen layer:
`q/k/v/o` projections + `gate/up/down` + 2 RMSNorms. MoE layers count all experts
(honest, even though the runner rejects MoE streaming).

**The #1 rule:** KV uses `num_key_value_heads` (GQA/MQA), never
`num_attention_heads`. Llama-3-70B has 8 KV heads vs 64 attention heads, an 8x
difference in cache size. Weight quantization shrinks `weight_bytes_per_param`
(int8 = 1.0, int4 = 0.5) but never the KV or activation bytes (we quantize weights
only).

## Tier selection (`tiering.py`, prompt §5)

```
tier0_peak = weights + kv@maxctx + activation + overhead
cache_dev  = floor((usable_device - resident - kv - activation - overhead) / per_layer)

if tier0_peak <= usable_device:                          -> Tier 0
elif weights <= usable_ram and cuda and not unified
        and cache_dev >= 2:                              -> Tier 1
elif weights <= usable_ram:                              -> Tier 2
else:                                                    -> Tier 3
```

`tier0_peak` includes KV at max context, so a budget that fits at load but would
OOM as KV grows is **not** chosen as Tier 0 (the headroom guard, surfaced up
front). If materialization OOMs for real, `_attach_with_demotion` catches it,
frees aggressively, and retries one tier down with a loud warning (demotion stays
within tiers 0/1/2, since demoting into Tier 3 needs on-disk shards).

## The streaming runner (`runner.py`, prompt §8)

The key correctness decision: **we never reimplement the model's forward.** In
transformers 5.x the decoder layer receives a precomputed `position_embeddings`
and the masks are built internally; reimplementing that per architecture is the #1
silent-wrongness risk. Instead each decoder layer gets:

- a `forward_pre_hook` that materializes its weights on the compute device (cache
  hit, or stage now) and pins it in the LRU, then hints the prefetcher,
- a `forward_hook` that releases the pin so the LRU may evict it.

The model's own attention/RoPE/KV-cache math runs untouched. This is what lets
`verify_against_full_load` pass: streamed logits match a plain full load bit-for-
bit (within float tolerance) for prefill and multi-step decode.

### Transfer modes

| Mode | When | Materialize | Evict |
|------|------|-------------|-------|
| `cuda` | discrete GPU | pinned-CPU master → async H2D on a dedicated stream, gated by a CUDA event | free the GPU copy |
| `copy` | CPU / test path | a real `clone()` per materialize | free the clone |
| `residency` | unified memory / MPS | no-op (weights already in the one pool) | no-op |

For Tier 3 the modes are the same, but the "master" is read from an mmap'd shard
each materialize instead of held in RAM, and `residency` is never used (Tier 3 was
chosen because the model does not fit RAM, so it must evict).

### LRU cache + prefetch (`cache.py`, `prefetch.py`)

The cache owns the policy (which layers are resident, eviction order, refcounts);
the `LayerHandle` owns weight movement and a small `absent → staging → ready`
state machine. Correctness invariants:

- **No use-after-free**: a layer pinned for a forward (or an in-flight prefetch)
  is never evicted (refcount guard). When every resident layer is pinned, the
  cache tolerates a one-layer transient overshoot rather than blocking compute.
- **Stream synchronization**: the compute stream waits on the prefetch stream's
  copy-complete CUDA event before using a freshly staged layer (events, not
  sleeps).
- **No deadlock**: the cache lock and a handle lock are never held simultaneously;
  the slow copy happens with neither lock held. The prefetch worker only stages
  into a free slot and never holds both locks.
- **Double buffering**: `prefetch_buffers >= 2` so layer N+1 stages during N's
  compute. The win comes from the H2D copy running on a CUDA stream / inside the
  GIL-releasing `.to(device)` C call, not from Python-level parallelism. On
  CPU/unified there is no real copy to overlap.

## Generation (`generation.py`, `model.py`, prompt §11)

A tier-agnostic decode loop drives `model(...)` with an externally held
`DynamicCache`. Logits processing order matches HF: repetition penalty →
temperature → top-k → top-p → min-p → softmax → sample. Batched decode uses
left-padding with `position_ids = cumsum(mask) - 1` and per-row stop tracking
(EOS + decode-aware stop strings); finished rows are masked and generation ends
when all rows finish. `stream=True` yields decoded pieces as produced.

## Shard format (`shard.py`, prompt §9)

```
<shard_dir>/
  layer_0000.safetensors   # one file per decoder layer, keys = param names
  ...
  resident.safetensors     # embed + final_norm + lm_head (tie-aware)
  config.json              # so reload needs no original checkpoint
  streamllm_manifest.json  # source id, dtype, quant scheme, per-shard bytes + sha256
```

Building checks free disk first (refuses with required-vs-available), is resumable
(skips shards whose hash matches the manifest), and validates each shard by
re-reading before committing it (atomic tmp→rename). `delete_source=True` removes
the original HF cache only after every shard is written and hash-verified.

Loading mmaps each shard via `safe_open` on demand. Reload currently instantiates
the model structure on CPU (real params + computed RoPE buffers) and frees decoder
weights to placeholders before streaming from disk, which briefly touches full-
model RAM; quantized shards reduce it. A fully meta-skeleton reload that recomputes
RoPE buffers without allocating any layer is future work.

## Quantization (`quant.py`, prompt §9)

Weight-only, per output channel. `int8` (1 B/param) and `int4` (nibble-packed,
~0.5 B/param) are pure torch, no extra dependency, so they are testable on CI.
`nf4` routes to bitsandbytes and raises a clear `pip install streamllm[quant]`
error if it is absent. Quantizing at shard time shrinks on-disk bytes, which is
the main Tier-3 speedup (fewer bytes read per layer). Dequantization happens per
layer at use time.

## Observability (`metrics.py`)

`RunMetrics` records tokens, TTFT, per-layer load-vs-compute time, cache hit rate,
bytes read, and peak memory. `.describe()` and the benchmark read it. The
load-vs-compute split is what lets the benchmark issue an honest I/O-vs-compute
verdict and report the effective disk read ceiling for Tier 3.
