# Troubleshooting

## OOM despite a tier choice

The memory model plans against *available* memory with a headroom factor, but
estimates are estimates. If you still OOM:

- **Lower `headroom`** (default 0.9). Try `StreamModel.from_pretrained(..., headroom=0.8)`
  or the env var `STREAMLLM_HEADROOM=0.8`. This plans to use less of the budget.
- **Lower `max_context`.** KV cache grows with `prompt_len + max_new_tokens`. A
  smaller context budget shrinks `kv_bytes` and can keep you in a higher tier.
- **Cap `cache_layers`.** For a streaming tier, `cache_layers=2` keeps only two
  decoder layers resident on the device at once (the minimum for double-buffered
  prefetch).
- **Quantize.** `quantize="int4"` cuts weight bytes ~4x vs fp16.
- streamllm also demotes one tier automatically on a real OOM at materialization,
  with a loud warning. If the bottom tier still fails, you get an
  `OutOfMemoryDemotionError` naming what to do.

Run `streamllm describe <model> --device <d> --max-context <n>` to see the full
budget breakdown without loading weights, and find the term that does not fit.

## Tier 3 is slow

It is **I/O-bound by physics**: every decoder layer is read from disk per token.
This is not a missing optimization. To reduce the wall (not remove it):

- **Quantize the shards** (`streamllm shard <model> --quantize int4`). Fewer bytes
  per layer means proportionally less read time.
- **Increase batch size.** On Tier 3 one layer read serves the whole batch, so
  throughput per token improves with batch. The benchmark's `--batch-sweep`
  demonstrates the curve.
- **Use a faster disk.** NVMe over SATA over network storage. `streamllm bench`
  prints the effective read GB/s you actually hit; compare it to your disk's spec.

If your model fits in RAM or VRAM, you should not be on Tier 3. Check
`streamllm describe` and consider quantizing to fit a higher tier.

## Unrecognized architecture

Module-graph discovery locates the decoder stack, embedding, final norm and LM
head by structure, not by name. If it fails on an unusual model:

```python
StreamModel.from_model(
    model, config,
    layer_module_path="transformer.h",                 # the decoder nn.ModuleList
    resident_module_paths={
        "embed": "transformer.wte",
        "norm": "transformer.ln_f",
        "lm_head": "lm_head",
    },
)
```

The error message names exactly which override to pass.

## MoE models

Mixture-of-Experts models (Mixtral, Qwen-MoE) are detected and **rejected** by
default, because expert-level streaming is not implemented and streaming the whole
(dense-equivalent) layer per token is extremely I/O-heavy. The error names the
expert count. To run anyway at high cost, pass `_allow_moe=True` to `from_model`.
`describe` still reports the (large) MoE numbers.

## MPS / CUDA caveats

- **MPS (Apple Silicon)** is treated as unified memory: streaming is residency
  management, not a real copy, and the prefetcher is a no-op. There is no pinned-
  memory H2D path on MPS.
- **Unified-memory CUDA (GB10, Jetson)** is detected when the device's total
  memory is within 20% of host RAM total. The available budget is reported as the
  min of GPU-free and host-available so the same bytes are not double-counted.
- **`torch.cuda.mem_get_info` can fail** on a memory-starved box (the CUDA context
  itself fails to initialize). streamllm catches this, logs a warning, and treats
  the machine as CPU-only rather than crashing. Free some GPU memory and retry if
  you expected CUDA.
- **Prefetch overlap only helps on a real device transfer.** On CPU/unified there
  is no copy to overlap with compute, so `cache_layers` and prefetch are about
  correctness and bookkeeping, not speed.

## trust_remote_code

Default `False`. Models that need custom modeling code require
`trust_remote_code=True`, which executes arbitrary code from the model repo.
streamllm warns when you opt in. Prefer safetensors checkpoints over pickle.

## Quantization without bitsandbytes

`quantize="int8"` and `quantize="int4"` are pure-torch and need no extra
dependency. `quantize="nf4"` needs bitsandbytes; if it is missing you get a clear
`pip install streamllm[quant]` message.
