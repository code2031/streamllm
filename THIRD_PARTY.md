# Third-party code, references, and licenses

streamllm is a clean reimplementation that **learned from** existing work rather
than vendoring it. No large verbatim blocks were copied. Where a concept or API
shaped the design, it is recorded below with the source and license. All listed
projects use permissive licenses (MIT or Apache-2.0) compatible with this
project's Apache-2.0 license.

## Runtime dependencies (not vendored, used as libraries)

| Project | License | Used for |
|---------|---------|----------|
| [PyTorch](https://github.com/pytorch/torch) | BSD-3-Clause | tensors, CUDA streams/events, the model forward |
| [transformers](https://github.com/huggingface/transformers) | Apache-2.0 | `AutoConfig`/`AutoModelForCausalLM`/`AutoTokenizer`, `DynamicCache`, the decoder layer modules whose forward we drive |
| [accelerate](https://github.com/huggingface/accelerate) | Apache-2.0 | `init_empty_weights()` for the meta-device skeleton used in tier estimation |
| [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | `safe_open` mmap/lazy load + `save_file` for shards |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | host RAM detection |
| [numpy](https://github.com/numpy/numpy) | BSD-3-Clause | misc numeric support |
| [tqdm](https://github.com/tqdm/tqdm) | MPL-2.0 / MIT | shard-build progress bar |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (optional) | MIT | the `nf4` quantization path only (the `int8`/`int4` paths are pure-torch, no dependency) |
| [FastAPI](https://github.com/fastapi/fastapi) + [uvicorn](https://github.com/encode/uvicorn) (optional, `web/` demo) | MIT / BSD-3-Clause | the playground server, which is a separate demo app, not part of the library |

## Concepts adapted (reimplemented, not copied)

- **Layer-by-layer weight streaming** — the core idea that an otherwise-too-large
  decoder can run by materializing one layer's weights at a time is the approach
  popularized by **[AirLLM](https://github.com/lyogavin/airllm)** (MIT). streamllm
  reimplements this with a different mechanism: forward hooks that swap a layer's
  `.data` in and out while the model's own forward (mask/RoPE/KV math) runs
  untouched, plus an LRU cache + a dedicated-stream prefetcher. No AirLLM source
  was used.
- **Meta-device skeleton + dispatch** — building a model with no allocated weights
  to inspect its structure, and materializing tensors on demand, mirrors the
  pattern in **accelerate**'s `init_empty_weights` / `dispatch_model` / disk
  offload (Apache-2.0). streamllm uses `init_empty_weights` directly and
  reimplements the materialization/eviction policy.
- **mmap weight loading** — the "let the OS page cache do the right thing" idea
  comes from **llama.cpp** (MIT) and **safetensors**. streamllm uses
  `safetensors.safe_open` rather than its own mmap.
- **Incremental KV-cache decode loop** — the position-ids / cache-position /
  left-padding handling for batched decoder-only generation follows the contract
  documented in **transformers** (Apache-2.0); streamllm drives the cache itself
  to integrate streaming + metrics.

## Attribution policy

If any future change vendors a non-trivial verbatim block from one of the above,
the upstream copyright/license header must be preserved in that file and the
source URL + what was adapted recorded here.
