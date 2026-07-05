"""describe (estimate-only, no weights), benchmark, and CLI wiring (prompt §13/§16)."""

from __future__ import annotations

import torch
from tests.conftest import make_hw, tiny_llama_config

from streamllm.benchmark import benchmark_model
from streamllm.model import StreamModel, estimate_only

BIG = dict(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=11008,
    num_hidden_layers=80,
    vocab_size=32000,
)


def test_estimate_only_returns_budget_without_weights():
    info = estimate_only(tiny_llama_config(), device="cpu")
    assert info["estimate_only"] is True
    assert "tier" in info and "budget" in info
    # A meta skeleton (zero memory) gives exact counts for known architectures.
    assert info["budget"]["source"] == "measured"
    assert info["budget"]["num_key_value_heads"] == 2


def test_estimate_only_accurate_for_gpt2():
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, GPT2Config

    # GPT-2 has a non-gated MLP (c_fc/c_proj), which the analytic SwiGLU formula
    # would mis-count. The meta-skeleton path must match the real layer count.
    cfg = GPT2Config(vocab_size=128, n_embd=32, n_layer=3, n_head=4, n_positions=64)
    info = estimate_only(cfg, device="cpu")
    assert info["budget"]["source"] == "measured"
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    actual = sum(p.numel() for p in m.transformer.h[0].parameters())
    assert info["budget"]["per_layer_params"] == actual


def test_estimate_only_big_model_streams_under_small_gpu():
    hw = make_hw(cuda=True, vram_total_gb=8, vram_free_gb=8, ram_avail_gb=64, unified=False)
    info = estimate_only(tiny_llama_config(**BIG), device="cuda", hardware=hw)
    assert info["tier"] in (1, 2, 3)
    assert info["budget"]["weights_gb"] > 8  # exceeds the 8 GB GPU


def _tiny_full(**kw):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = tiny_llama_config(**kw)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    return StreamModel.from_model(model, cfg, tier="full", device="cpu"), cfg


def test_benchmark_tier0_compute_bound():
    sm, cfg = _tiny_full()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    res = benchmark_model(sm, ids, max_new_tokens=8, trials=3, warmup=1)
    assert res.tier == 0
    assert "compute-bound" in res.verdict
    assert res.tokens_per_s_median >= 0
    # median and p90 both present
    assert res.tokens_per_s_p90 >= 0


def test_benchmark_batch_sweep_on_streamed():
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = tiny_llama_config(num_hidden_layers=4)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    sm = StreamModel.from_model(model, cfg, tier="ram", device="cpu", cache_layers=2)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    res = benchmark_model(sm, ids, max_new_tokens=6, trials=2, warmup=1, batch_sizes=[1, 2, 4])
    assert len(res.batch_sweep) == 3
    assert {r["batch_size"] for r in res.batch_sweep} == {1, 2, 4}
    assert res.tier == 2


def test_benchmark_json_csv_roundtrip(tmp_path):
    sm, cfg = _tiny_full()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    res = benchmark_model(sm, ids, max_new_tokens=6, trials=2, warmup=0)
    res.write_json(tmp_path / "b.json")
    res.write_csv(tmp_path / "b.csv")
    import json

    data = json.loads((tmp_path / "b.json").read_text())
    assert data["tier"] == 0 and "verdict" in data
    assert (tmp_path / "b.csv").read_text().count("\n") >= 2  # header + row


def test_cli_describe_json(monkeypatch, capsys):
    import transformers

    from streamllm.cli import main

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: tiny_llama_config()),
    )
    rc = main(["describe", "dummy/model", "--device", "cpu", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    import json

    info = json.loads(out)
    assert info["estimate_only"] is True


def test_cli_describe_human(monkeypatch, capsys):
    import transformers

    from streamllm.cli import main

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: tiny_llama_config(**BIG)),
    )
    rc = main(["describe", "dummy/model", "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would pick" in out and "budget (peak)" in out
