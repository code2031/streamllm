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
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "the playground needs FastAPI + uvicorn. Install with: pip install 'streamllm[web]'"
    ) from exc

from streamllm.logging_utils import get_logger

_log = get_logger("web")
_WEB = Path(__file__).parent

app = FastAPI(title="streamllm playground")

_MODEL: Any = None
_MODEL_LOCK = asyncio.Lock()


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
    return sm.describe()


@app.post("/api/generate")
async def generate(request: Request) -> StreamingResponse:
    body = await request.json()
    prompt = str(body.get("prompt", ""))
    max_new = int(body.get("max_new_tokens", 128))
    temperature = float(body.get("temperature", 0.8))
    do_sample = bool(body.get("do_sample", True))
    sm = await _ensure_model()

    async def event_stream() -> Any:
        # Run the (blocking) generator in a thread, forwarding pieces over SSE.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

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
            except Exception as exc:  # surface generation errors to the client
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
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

        asyncio.get_running_loop().run_in_executor(None, produce)
        while True:
            kind, payload = await queue.get()
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            if kind in ("done", "error"):
                break

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
