#!/usr/bin/env python3
"""Standalone benchmark runner (prompt §16).

Thin wrapper over ``streamllm bench`` so the benchmark is runnable as a script:

    python bench/benchmark.py meta-llama/Llama-3.1-8B \\
        --prompt "Explain streaming inference" --max-new-tokens 64 \\
        --trials 5 --batch-sweep 1,2,4,8 --json out.json --csv out.csv

It reports the chosen tier + deciding numbers, median/p90 tokens/sec, TTFT,
decode tok/s, per-layer load-vs-compute, cache hit rate, peak memory, and a
one-line I/O-vs-compute verdict. JSON + CSV carry enough metadata to reproduce.
"""

from __future__ import annotations

import sys

from streamllm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["bench", *sys.argv[1:]]))
