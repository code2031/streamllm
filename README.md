# streamllm

**Speed-first, auto-tiering inference for large language models on whatever
hardware you have.** The headline feature is the *policy that decides how little
to stream* — not streaming itself.

streamllm detects your memory budget and picks the **least-streaming tier that
fits**. When the model fits in your accelerator, it does **not** stream — it
loads normally and tells you a dedicated engine would be faster. When the model
doesn't fit, it falls back through tiers that stream as little as possible, and
it is **honest by construction** about what that costs.

> The bottom tier (disk-backed streaming) is **I/O-bound by physics**: reading
> every layer per token is disk-bandwidth-limited. The mitigations here (quantized
> shards, batch amortization, prefetch overlap) reduce the wall; none remove it.
> The code says so in logs and reports rather than implying otherwise.

## The tier table (read this first)

| Tier | Name | When it's chosen | What happens | Realistic expectation |
|------|------|------------------|--------------|-----------------------|
| **0** | `full` | Weights + KV-at-max-context + activations fit the accelerator | Plain full load, **no streaming** | Use **vLLM / TGI / llama.cpp** — they'll be faster. streamllm adds no value here, and says so. |
| **1** | `gpu_ram` | Doesn't fit VRAM, fits host RAM, ≥2 layers resident on a discrete GPU | Keep embeddings/norm/head/KV + as many decoder layers as fit on GPU; stream the rest from **pinned host RAM** with LRU + async prefetch | Slower than Tier 0, usually compute-bound if PCIe keeps up |
| **2** | `ram` | Fits host RAM, little/no usable GPU, or **unified memory** | All decoder weights live in RAM; stream layer-by-layer into compute. On unified memory this is residency management, not a real copy | Workable; on unified memory close to native |
| **3** | `disk` | Doesn't fit RAM | Stream layers from **mmap'd on-disk shards** | **I/O-bound by physics.** The benchmark reports the effective disk read ceiling you actually hit. |

`tier="auto"` (default) picks for you and **logs the numbers that forced the
choice**. Override with `tier=0|1|2|3` or `tier="full"|"gpu_ram"|"ram"|"disk"`.

## Quickstart

```python
from streamllm import StreamModel

sm = StreamModel.from_pretrained("meta-llama/Llama-3.1-8B", tier="auto")
print(sm.decision.summary())          # which tier, and why (the numbers)

for piece in sm.generate("Streaming inference is", max_new_tokens=64, stream=True):
    print(piece, end="", flush=True)   # feel the tier's speed
```

Plan a model on a machine that **can't run it** — no weights loaded:

```bash
streamllm describe meta-llama/Llama-3.1-70B --device cuda --quantize int4
```

```
would pick:   Tier 1 (gpu_ram, backing=ram, device=cuda:0, cache_layers=18)
reason:       cache_layers=floor((usable_vram 21.6GB - resident 0.27 - KV 1.25 - ...
budget (peak):
  weights      37.40 GB (80 layers x 0.46 GB/layer + resident 0.27 GB)
  KV @ ctx=8192  1.25 GB  (kv_heads=8)
  activation   0.10 GB
```

## CLI

```bash
streamllm run <model> --prompt "..." [--max-new-tokens N] [--tier auto] [--quantize int4] [--stream]
streamllm shard <model> [--out PATH] [--quantize int4] [--delete-source]
streamllm bench <model> [--trials 5] [--batch-sweep 1,2,4,8] [--json out.json] [--csv out.csv]
streamllm describe <model>      # estimate-only: the tier + memory math, no weights
```

## Hardware-fit matrix (expected tier)

Approximate tier for a single accelerator + system RAM, at `max_context≈8k`. Your
mileage varies with context length and headroom; run `streamllm describe` for the
real numbers on your box.

| Model (params) | fp16 size | 8 GB GPU + 16 GB RAM | 24 GB GPU + 64 GB RAM | 128 GB unified | CPU-only 32 GB |
|----------------|-----------|----------------------|------------------------|----------------|-----------------|
| 7B  | ~14 GB | Tier 1 | **Tier 0** | **Tier 0** | Tier 0 |
| 13B | ~26 GB | Tier 1 | **Tier 0** | **Tier 0** | Tier 2/3 |
| 70B fp16 | ~140 GB | Tier 3 | Tier 1 | Tier 2 | Tier 3 |
| 70B int4 | ~37 GB | Tier 1 | **Tier 0** | **Tier 0** | Tier 2 |
| 405B int4 | ~210 GB | Tier 3 | Tier 3 | Tier 3 | Tier 3 |

Bold = "don't stream, use a real engine." Tier 3 cells are I/O-bound.

## Public API

```python
StreamModel.from_pretrained(model_id, *, tier="auto", dtype=None, quantize=None,
    cache_layers="auto", max_context=None, headroom=0.9, device="auto",
    shard_path=None, trust_remote_code=False, on_context_overflow="error")

.generate(input_ids|str|list[str], *, max_new_tokens, do_sample=False,
    temperature=1.0, top_k=None, top_p=None, min_p=None, repetition_penalty=1.0,
    stop=None, seed=None, stream=False)        # stream=True yields tokens

.forward(input_ids, ...) -> logits
.describe() -> dict       # hardware, tier + deciding numbers, budget breakdown
.shard(out_path=..., quantize=...) / streamllm.shard_model(...)
StreamModel.from_shards(shard_dir)             # reload from disk shards

streamllm.estimate_only(model_id, ...) -> dict # the describe math, no weights
```

## Installation

```bash
pip install streamllm                 # core (no bitsandbytes)
pip install "streamllm[quant]"        # + bitsandbytes for nf4
pip install "streamllm[test,dev]"     # pytest/hypothesis, ruff/mypy/pre-commit
```

Python 3.11+. Core deps: `torch`, `transformers`, `accelerate`, `safetensors`,
`psutil`, `numpy`, `tqdm`. Importing the core library does **not** require
`bitsandbytes`.

### Compatibility matrix

| Axis | Tested | Notes |
|------|--------|-------|
| Python | 3.11, 3.12 | `>=3.11` required (PEP 604 types, `from __future__`) |
| torch | 2.1 - 2.11 | dev box: 2.11+cu130 |
| transformers | 4.40 - 5.3 | 5.x decode path (precomputed `position_embeddings`, `DynamicCache`) is the dev target |
| CUDA | discrete + unified (GB10) | `mem_get_info` failures degrade to CPU |
| MPS | treated as unified memory | no pinned-memory H2D path |
| CPU-only | yes | full test suite runs here, no GPU, no downloads |
| OS | Linux primary; macOS via MPS/CPU | Windows untested |
| quant | `int8`/`int4` pure-torch; `nf4` needs `bitsandbytes` | core import never needs bitsandbytes |

The full pytest suite runs on **CPU with no model downloads** using tiny random
configs and mocked memory budgets, so CI needs no GPU.

## Non-goals (deliberately out of scope for v1)

- **Training / fine-tuning.** Inference only.
- **Multi-GPU tensor/pipeline parallelism.** Single accelerator (or CPU). A second
  GPU may serve only as additional offload memory.
- **Serving infra.** No HTTP server, no continuous-batching scheduler, no
  paged-attention allocator. A synchronous `.generate` + CLI is the surface. (The
  `web/` playground is a separate demo app, not part of the library.)
- **Speculative decoding, draft models, cross-request prompt caching.**
- **MoE streaming optimization.** MoE models are *detected* and either run (dense-
  equivalent layer streaming) or rejected with a clear message — never silently
  mis-handled.

## Limitations (first, not in fine print)

- **Tier 0 means "use a real engine."** If your model fits, vLLM/TGI/llama.cpp are
  faster. streamllm still runs, but it tells you.
- **Tier 3 is I/O-bound by physics.** Quantize, increase batch size, or use a
  faster disk. None of these remove the per-token full-weight read.
- **Memory estimates are estimates.** The headroom factor (default 0.9) guards
  against "fits at load, OOMs as KV grows"; lower it if you still OOM.
- **Tier-3 reload** currently instantiates the model structure on CPU to obtain
  computed RoPE buffers, then frees decoder weights before streaming from disk —
  so reload briefly touches full-model RAM. Quantized shards reduce it.
- **Prefetch overlap** only yields wall-clock wins with a real device transfer
  (CUDA). On CPU/unified memory there is no copy to overlap.

See [docs/architecture.md](docs/architecture.md) for the runner, prefetch/eviction,
shard format and the memory model formulas, and
[docs/troubleshooting.md](docs/troubleshooting.md) for OOM/slow-Tier-3/unknown-
architecture fixes. Adapted code + licenses are recorded in
[THIRD_PARTY.md](THIRD_PARTY.md).
