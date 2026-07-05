"""Playground server smoke tests + the security hardening (web/server.py).

Loads web/server.py (not a package) via importlib, drives it with FastAPI's
TestClient against the tiny offline demo model. Also asserts the resource guards:
max_new_tokens is clamped and an oversized prompt is rejected.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient needs httpx


def _load_server(monkeypatch, **env):
    monkeypatch.setenv("STREAMLLM_DEVICE", "cpu")
    monkeypatch.delenv("STREAMLLM_MODEL", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    path = pathlib.Path(__file__).resolve().parent.parent / "web" / "server.py"
    spec = importlib.util.spec_from_file_location("streamllm_web_server_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _done_payload(sse_text: str) -> dict:
    for block in sse_text.split("\n\n"):
        if "event: done" in block:
            return json.loads(block.split("data: ", 1)[1])
    raise AssertionError(f"no done event in SSE:\n{sse_text[:500]}")


def test_describe_returns_a_tier(monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_server(monkeypatch)
    client = TestClient(mod.app)
    r = client.get("/api/describe")
    assert r.status_code == 200
    d = r.json()
    assert d["tier"] in (0, 1, 2, 3)
    assert "budget" in d


def test_generate_sse_completes(monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_server(monkeypatch)
    client = TestClient(mod.app)
    r = client.post("/api/generate", json={"prompt": "hi", "max_new_tokens": 6})
    assert r.status_code == 200
    _done_payload(r.text)  # a well-formed stream always ends with a done event


def test_generate_clamps_max_new_tokens(monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_server(monkeypatch, STREAMLLM_WEB_MAX_NEW_TOKENS="6")
    client = TestClient(mod.app)
    r = client.post("/api/generate", json={"prompt": "hi", "max_new_tokens": 10_000_000})
    assert r.status_code == 200
    payload = _done_payload(r.text)
    assert payload["tokens"] <= 6  # clamped, not a million


def test_generate_rejects_oversized_prompt(monkeypatch):
    from fastapi.testclient import TestClient

    mod = _load_server(monkeypatch, STREAMLLM_WEB_MAX_PROMPT_CHARS="100")
    client = TestClient(mod.app)
    r = client.post("/api/generate", json={"prompt": "x" * 500})
    assert r.status_code == 413
