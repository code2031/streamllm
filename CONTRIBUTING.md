# Contributing to streamllm

Thanks for helping. This is a correctness-first library; the bar for the streaming
path, the tiering math, and the KV-cache loop is high because those fail silently.

## Setup

```bash
pip install -e ".[test,dev]"      # core + pytest/hypothesis + ruff/mypy/pre-commit
pre-commit install                # optional: run ruff+mypy on commit
```

Python 3.11+ is required.

## The gates (all must pass before a PR)

```bash
PYTHONPATH=src python3 -m pytest -q            # CPU-only, no downloads
ruff check src tests bench web/server.py
ruff format --check src tests bench
mypy src
```

CI runs exactly these on Python 3.11 and 3.12. The suite uses tiny random-weight
configs and mocked memory budgets, so it needs no GPU and no network.

## What to know before you touch the hot paths

- **`tests/test_verify.py` is the oracle.** Streamed logits must match a plain full
  load for prefill and multi-step decode. If you change `runner.py`, `cache.py`,
  `prefetch.py`, or the generate loop in `model.py`, keep it green.
- **The tiering math is pure.** `select_tier` does no torch allocation, so
  `tests/test_tiering.py` feeds mocked budgets. Add a case there for any policy change.
- **KV uses `num_key_value_heads`, never `num_attention_heads`.** This is the #1
  estimation bug. `tests/test_memory.py` guards it.
- **Do not reimplement the model forward.** transformers 5.x precomputes
  `position_embeddings` and builds masks internally. Stream weights via hooks so the
  model's own math runs. Reimplementing it is the fastest way to silent wrongness.

## Conventions

- `from __future__ import annotations`, full type hints, Google/NumPy docstrings on
  public API.
- Log under the `streamllm` logger; never `print` except in `cli.py`.
- Every automatic decision is logged with the numbers that forced it, and is
  overridable.
- Honest by construction: do not soften the Tier 0 ("use a real engine") or Tier 3
  ("I/O-bound by physics") messages.
- No em-dashes in the web copy (`DESIGN.md` is the frontend design system).

## Commits and PRs

- Conventional-commit style prefixes (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- One logical change per PR. Update `CHANGELOG.md` under `[Unreleased]`.
- New behavior needs a test. New tiers/policies need a `test_tiering.py` case and,
  if they touch streaming, a `test_verify.py` pass.
