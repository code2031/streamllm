"""``streamllm`` console entry point (prompt §13).

Subcommands:
* ``run``      — generate, streaming tokens to stdout, tier + budget to stderr.
* ``shard``    — build resumable disk shards with a free-space check + progress.
* ``bench``    — run the benchmark (§16) and write JSON/CSV.
* ``describe`` — show the tier it *would* pick and the memory math, **without
  loading weights** (estimate-only), so users can plan on any machine.

This is the only place we ``print`` (user-facing output); everything else logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .logging_utils import get_logger, set_level

_log = get_logger("cli")


def _gb(x: float) -> str:
    return f"{x / 1e9:.2f} GB"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="HF model id or local path (or shard dir for run)")
    p.add_argument("--tier", default="auto", help="auto|0|1|2|3|full|gpu_ram|ram|disk")
    p.add_argument("--device", default="auto", help="auto|cuda|cuda:0|mps|cpu")
    p.add_argument("--dtype", default=None, help="float16|bfloat16|float32")
    p.add_argument("--quantize", default=None, choices=[None, "int8", "int4", "nf4"])
    p.add_argument("--max-context", type=int, default=None)
    p.add_argument("--headroom", type=float, default=0.9)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(prog="streamllm", description=__doc__)
    parser.add_argument("--version", action="version", version=f"streamllm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate text, streaming to stdout")
    _add_common(run)
    run.add_argument("--prompt", required=True)
    run.add_argument("--max-new-tokens", type=int, default=128)
    run.add_argument("--stream", action="store_true", help="stream tokens as produced")
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-k", type=int, default=None)
    run.add_argument("--top-p", type=float, default=None)
    run.add_argument("--do-sample", action="store_true")
    run.add_argument("--seed", type=int, default=None)
    run.set_defaults(func=_cmd_run)

    shard = sub.add_parser("shard", help="build disk shards (resumable)")
    shard.add_argument("model")
    shard.add_argument("--out", default=None, help="output shard dir")
    shard.add_argument("--dtype", default=None)
    shard.add_argument("--quantize", default=None, choices=[None, "int8", "int4", "nf4"])
    shard.add_argument(
        "--delete-source",
        action="store_true",
        help="remove the HF cache AFTER all shards verify (opt-in)",
    )
    shard.add_argument("--trust-remote-code", action="store_true")
    shard.add_argument("-v", "--verbose", action="store_true")
    shard.set_defaults(func=_cmd_shard)

    bench = sub.add_parser("bench", help="benchmark tokens/sec, TTFT, I/O verdict")
    _add_common(bench)
    bench.add_argument("--prompt", default="The quick brown fox")
    bench.add_argument("--max-new-tokens", type=int, default=32)
    bench.add_argument("--trials", type=int, default=3)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument(
        "--batch-sweep", default=None, help="comma-separated batch sizes, e.g. 1,2,4,8"
    )
    bench.add_argument("--json", default=None, help="write JSON here")
    bench.add_argument("--csv", default=None, help="write CSV here")
    bench.set_defaults(func=_cmd_bench)

    describe = sub.add_parser("describe", help="show the tier + memory math (no weights)")
    _add_common(describe)
    describe.add_argument("--batch-size", type=int, default=1)
    describe.add_argument("--json", action="store_true", help="emit raw JSON")
    describe.set_defaults(func=_cmd_describe)

    return parser


def _maybe_verbose(args: argparse.Namespace) -> None:
    if getattr(args, "verbose", False):
        set_level("DEBUG")


def _cmd_describe(args: argparse.Namespace) -> int:
    _maybe_verbose(args)
    from .model import estimate_only

    info = estimate_only(
        args.model,
        device=args.device,
        tier=args.tier,
        dtype=args.dtype,
        quantize=args.quantize,
        max_context=args.max_context,
        headroom=args.headroom,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
    )
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    _print_describe(info)
    return 0


def _print_describe(info: dict[str, Any]) -> None:
    b = info["budget"]
    print(f"model:        {info['model']}")
    print(
        f"would pick:   Tier {info['tier']} ({info['tier_name']}, backing={info['backing']}, "
        f"device={info['compute_device']}, cache_layers={info['cache_layers']})"
    )
    print(f"reason:       {info['reason']}")
    if info.get("honest_note"):
        print(f"note:         {info['honest_note']}")
    if info.get("is_moe"):
        print(
            f"WARNING:      MoE model ({info['num_experts']} experts/layer); "
            "expert-level streaming is not implemented"
        )
    hw = info["hardware"]
    print(
        f"hardware:     cuda={hw['cuda']} devices={hw['devices']} unified={hw['unified_memory']} "
        f"ram_avail={hw['ram_avail_gb']}GB disk_free={hw['disk_free_gb']}GB"
    )
    print("budget (peak):")
    print(
        f"  weights      {b['weights_gb']} GB ({b['n_layers']} layers x "
        f"{b['per_layer_gb']} GB/layer + resident {b['resident_gb']} GB)"
    )
    print(f"  KV @ ctx={b['context']}  {b['kv_gb']} GB  (kv_heads={b['num_key_value_heads']})")
    print(f"  activation   {b['activation_gb']} GB")
    print(f"  source       {b['source']} param counts")


def _cmd_run(args: argparse.Namespace) -> int:
    _maybe_verbose(args)
    from .model import StreamModel

    sm = StreamModel.from_pretrained(
        args.model,
        tier=args.tier,
        device=args.device,
        dtype=args.dtype,
        quantize=args.quantize,
        max_context=args.max_context,
        headroom=args.headroom,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"[streamllm] {sm.decision.summary()} :: {sm.decision.reason}", file=sys.stderr)
    if sm.decision.honest_note:
        print(f"[streamllm] {sm.decision.honest_note}", file=sys.stderr)

    gen_kw = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    if args.stream:
        for piece in sm.generate(args.prompt, stream=True, **gen_kw):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        print(sm.generate(args.prompt, **gen_kw))
    m = sm.last_metrics
    if m is not None:
        print(
            f"[streamllm] {m.generated_tokens} tok in {m.total_s:.2f}s "
            f"({m.tokens_per_s:.1f} tok/s, TTFT {m.ttft_s or 0:.3f}s, "
            f"{m.bottleneck()})",
            file=sys.stderr,
        )
    return 0


def _cmd_shard(args: argparse.Namespace) -> int:
    _maybe_verbose(args)
    from transformers import AutoConfig, AutoModelForCausalLM

    from .config import StreamConfig
    from .model import _resolve_dtype
    from .shard import shard_model

    cfg = StreamConfig.from_env()
    out = args.out or str(cfg.cache_dir / ("shards--" + args.model.replace("/", "--")))
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print(f"[streamllm] loading {args.model} to shard -> {out}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=_resolve_dtype(args.dtype, config),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    res = shard_model(
        model,
        config,
        out_path=out,
        dtype=_resolve_dtype(args.dtype, config),
        quantize=args.quantize,
        delete_source=args.delete_source,
        source_model=args.model,
        progress=True,
    )
    print(
        f"[streamllm] sharded {len(res.manifest['layers'])} layers "
        f"({_gb(res.total_bytes)}); built={len(res.built)} skipped={len(res.skipped)}"
    )
    print(f"[streamllm] manifest: {res.manifest_path}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    _maybe_verbose(args)
    from .benchmark import benchmark_model
    from .model import StreamModel

    sm = StreamModel.from_pretrained(
        args.model,
        tier=args.tier,
        device=args.device,
        dtype=args.dtype,
        quantize=args.quantize,
        max_context=args.max_context,
        headroom=args.headroom,
        trust_remote_code=args.trust_remote_code,
    )
    batch_sizes = [int(x) for x in args.batch_sweep.split(",")] if args.batch_sweep else None
    res = benchmark_model(
        sm,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        trials=args.trials,
        warmup=args.warmup,
        batch_sizes=batch_sizes,
    )
    print(f"[streamllm] {res.verdict}", file=sys.stderr)
    print(
        f"tokens/s: median={res.tokens_per_s_median} p90={res.tokens_per_s_p90}  "
        f"TTFT={res.ttft_s_median}s  decode={res.decode_tokens_per_s_median} tok/s"
    )
    print(
        f"layer load/compute (mean): {res.layer_load_s_mean}s / {res.layer_compute_s_mean}s  "
        f"hit_rate={res.cache_hit_rate}  io_fraction={res.io_fraction}"
    )
    if res.batch_sweep:
        print(
            "batch sweep (tokens/s):",
            ", ".join(f"b{r['batch_size']}={r['tokens_per_s']}" for r in res.batch_sweep),
        )
    if args.json:
        res.write_json(args.json)
        print(f"[streamllm] wrote {args.json}", file=sys.stderr)
    if args.csv:
        res.write_csv(args.csv)
        print(f"[streamllm] wrote {args.csv}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[streamllm] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # surface actionable errors, not tracebacks, by default
        if getattr(args, "verbose", False):
            raise
        print(f"[streamllm] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
