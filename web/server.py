"""streamllm playground server (separate demo app, NOT part of the library).

A thin FastAPI wrapper over ``StreamModel.generate`` with SSE token streaming and
a ``/api/describe`` endpoint, plus static serving of the landing page + playground.
The library spec deliberately ships no server (prompt §3); this lives under
``web/`` so the no-server-in-library boundary holds.

Run:
    pip install "streamllm[web]"
    STREAMLLM_MODEL=meta-llama/Llama-3.1-8B python web/server.py   # or:
    uvicorn web.server:app --port 6909

Env:
    STREAMLLM_MODEL   HF id / local path / shard dir (default: a tiny demo config)
    STREAMLLM_TIER    auto|0|1|2|3 (default auto)
    STREAMLLM_DEVICE  auto|cuda|cpu (default auto)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "the playground needs FastAPI + uvicorn. Install with: pip install 'streamllm[web]'"
    ) from exc

from streamllm.logging_utils import get_logger

_log = get_logger("web")
_WEB = Path(__file__).parent

# Resource-exhaustion guards for a playground that might get exposed. Overridable
# by the operator via env, but bounded by default so an untrusted client cannot
# request a million-token generation or a giant prompt, and cannot pile up
# concurrent generations against the single shared model.
MAX_NEW_TOKENS = int(os.environ.get("STREAMLLM_WEB_MAX_NEW_TOKENS", "512"))
MAX_PROMPT_CHARS = int(os.environ.get("STREAMLLM_WEB_MAX_PROMPT_CHARS", "8000"))

app = FastAPI(title="streamllm playground")

_MODEL: Any = None
_MODEL_LOCK = asyncio.Lock()
# Only one generation at a time: the shared streaming runner's cache/metrics are
# not safe for concurrent .generate calls. Extra requests get 429 (see below).
_GEN_SEMA = asyncio.Semaphore(1)


def _safe_model_name(name: Any) -> Any:
    """Avoid leaking an absolute local filesystem path in ``/api/describe``."""
    if isinstance(name, str) and ("/" in name or "\\" in name):
        with contextlib.suppress(Exception):
            if Path(name).exists():
                return Path(name).name
    return name


def _load_model() -> Any:
    """Build the StreamModel once. A tiny random model is used if none is set."""
    from streamllm.model import StreamModel

    model_id = os.environ.get("STREAMLLM_MODEL")
    tier = os.environ.get("STREAMLLM_TIER", "auto")
    device = os.environ.get("STREAMLLM_DEVICE", "auto")
    if model_id:
        _log.info("playground loading %s (tier=%s, device=%s)", model_id, tier, device)
        return StreamModel.from_pretrained(model_id, tier=tier, device=device)

    # No model configured: a tiny random-weight demo so the UI works offline.
    _log.warning("STREAMLLM_MODEL not set; using a tiny random demo model (gibberish output)")
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

    cfg = LlamaConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    tok = None
    with contextlib.suppress(Exception):  # pragma: no cover - offline
        tok = AutoTokenizer.from_pretrained("gpt2")
    return StreamModel.from_model(model, cfg, tier=tier, device=device, tokenizer=tok)


async def _ensure_model() -> Any:
    global _MODEL
    if _MODEL is None:
        async with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = await asyncio.to_thread(_load_model)
    return _MODEL


@app.get("/api/describe")
async def describe() -> dict[str, Any]:
    sm = await _ensure_model()
    info = sm.describe()
    info["model"] = _safe_model_name(info.get("model"))
    return info


@app.post("/api/generate")
async def generate(request: Request) -> StreamingResponse:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    prompt = str(body.get("prompt", ""))
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail=f"prompt exceeds {MAX_PROMPT_CHARS} chars")
    # Clamp untrusted numeric knobs to bounded ranges.
    max_new = max(1, min(int(body.get("max_new_tokens", 128)), MAX_NEW_TOKENS))
    temperature = min(max(float(body.get("temperature", 0.8)), 0.0), 5.0)
    do_sample = bool(body.get("do_sample", True))

    # One generation at a time; reject rather than queue so load cannot pile up.
    if _GEN_SEMA.locked():
        raise HTTPException(status_code=429, detail="busy: one generation at a time")
    sm = await _ensure_model()

    async def event_stream() -> Any:
        # Run the (blocking) generator in a thread, forwarding pieces over SSE.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        await _GEN_SEMA.acquire()

        def produce() -> None:
            try:
                for piece in sm.generate(
                    prompt,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=temperature,
                    stream=True,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", piece))
            except Exception:
                # Log the detail server-side; never leak internals to the client.
                _log.exception("playground generation failed")
                loop.call_soon_threadsafe(queue.put_nowait, ("error", "generation failed"))
            finally:
                m = sm.last_metrics
                done = {
                    "tokens": m.generated_tokens if m else 0,
                    "tokens_per_s": round(m.tokens_per_s, 1) if m else 0,
                    "ttft_s": round(m.ttft_s or 0, 3) if m else 0,
                    "tier": sm.decision.tier,
                    "bottleneck": m.bottleneck() if m else "unknown",
                }
                loop.call_soon_threadsafe(queue.put_nowait, ("done", done))

        try:
            loop.run_in_executor(None, produce)
            while True:
                kind, payload = await queue.get()
                yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
                # `done` is always the terminal event (produce emits it in finally,
                # even after an error), so the stream always ends cleanly.
                if kind == "done":
                    break
        finally:
            _GEN_SEMA.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


@app.get("/playground/")
async def playground() -> FileResponse:
    return FileResponse(_WEB / "playground" / "index.html")


# Static assets (styles.css, calc.js, playground/app.js).
app.mount("/", StaticFiles(directory=str(_WEB)), name="static")


def main() -> None:  # pragma: no cover - entry helper
    import uvicorn

    port = int(os.environ.get("STREAMLLM_WEB_PORT", "6909"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
