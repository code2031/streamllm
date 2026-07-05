"""``StreamModel`` — the public entry point (prompt §12).

Holds a (possibly streaming) HF causal LM plus the tier decision and metrics. The
generate loop is tier-agnostic: it calls ``self.model(...)`` with an externally
held :class:`~transformers.DynamicCache`; whether decoder weights stream in via
forward hooks (Tiers 1/2/3) or are fully resident (Tier 0) is invisible here.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .config import StreamConfig
from .errors import ContextOverflowError, OutOfMemoryDemotionError, UnsupportedModelError
from .generation import SamplingParams, StopController, sample_next
from .graph import ModelGraph, discover_graph
from .hardware import HardwareInfo, detect_hardware
from .logging_utils import get_logger
from .memory import MemoryEstimate, count_params_from_config, estimate_memory
from .metrics import RunMetrics
from .tiering import TierDecision, demotion_ladder, select_tier

if TYPE_CHECKING:
    from .runner import StreamingRunner

_log = get_logger("model")


def _measured_param_counts(graph: ModelGraph) -> tuple[int, int]:
    """Exact per-layer and resident param counts from the (meta-ok) modules."""
    per_layer = sum(p.numel() for p in graph.layers[0].parameters())
    embed_ids = {id(p) for p in graph.embed_tokens.parameters()}
    resident = sum(p.numel() for p in graph.embed_tokens.parameters())
    resident += sum(p.numel() for p in graph.final_norm.parameters())
    # Don't double-count tied lm_head weight (it is the embedding tensor).
    resident += sum(p.numel() for p in graph.lm_head.parameters() if id(p) not in embed_ids)
    return int(per_layer), int(resident)


class StreamModel:
    """A model that runs at the least-streaming tier its hardware allows."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: Any,
        tokenizer: Any,
        graph: ModelGraph,
        decision: TierDecision,
        hardware: HardwareInfo,
        stream_config: StreamConfig,
        estimate: MemoryEstimate,
        device: str,
        max_context: int,
        on_context_overflow: str = "error",
        runner: StreamingRunner | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.graph = graph
        self.decision = decision
        self.hardware = hardware
        self.stream_config = stream_config
        self.estimate = estimate
        self.device = device
        self.max_context = max_context
        self.on_context_overflow = on_context_overflow
        self.runner = runner
        self.last_metrics: RunMetrics | None = None
        self.model.eval()

    # --------------------------------------------------------------- builders

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        config: Any,
        *,
        tokenizer: Any = None,
        tier: object = "auto",
        device: str = "auto",
        dtype: object | None = None,
        quantize: str | None = None,
        cache_layers: object = "auto",
        max_context: int | None = None,
        headroom: float = 0.9,
        stream_config: StreamConfig | None = None,
        hardware: HardwareInfo | None = None,
        on_context_overflow: str = "error",
        layer_module_path: str | None = None,
        resident_module_paths: dict[str, str] | None = None,
        allow_demotion: bool = True,
        _allow_moe: bool = False,
    ) -> StreamModel:
        """Wrap an in-memory model (used by tests, verify, and from_pretrained)."""
        cfg = StreamConfig.from_env(stream_config or StreamConfig(headroom=headroom))
        hw = hardware or detect_hardware(cfg.cache_dir)
        compute_device = hw.resolve_device(device)
        graph = discover_graph(
            model,
            config,
            layer_module_path=layer_module_path,
            resident_module_paths=resident_module_paths,
        )
        max_ctx = max_context or _infer_max_context(config, cfg)
        est = _build_estimate(config, graph, cfg, dtype, quantize, max_ctx)

        if est.is_moe and not _allow_moe:
            raise UnsupportedModelError(
                f"model is MoE with {est.num_experts} experts/layer; expert-level "
                "streaming is not implemented. Pass _allow_moe=True to stream the "
                "whole (dense-equivalent) layer per token at high I/O cost."
            )

        decision = select_tier(hw, est, cfg, device=device, tier_override=tier)
        _apply_cache_layers_override(decision, cache_layers)

        decision, runner = _attach_with_demotion(
            model,
            graph,
            est,
            cfg,
            hw,
            compute_device,
            device,
            decision,
            cache_layers=cache_layers,
            allow_demotion=allow_demotion,
        )
        return cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            graph=graph,
            decision=decision,
            hardware=hw,
            stream_config=cfg,
            estimate=est,
            device=compute_device,
            max_context=max_ctx,
            on_context_overflow=on_context_overflow,
            runner=runner,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        tier: object = "auto",
        dtype: object | None = None,
        quantize: str | None = None,
        cache_layers: object = "auto",
        max_context: int | None = None,
        headroom: float = 0.9,
        device: str = "auto",
        shard_path: str | None = None,
        trust_remote_code: bool = False,
        on_context_overflow: str = "error",
        stream_config: StreamConfig | None = None,
    ) -> StreamModel:
        """Load a model from an HF id/path and select a tier (prompt §12)."""
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if trust_remote_code:
            _log.warning(
                "trust_remote_code=True executes arbitrary model code from %s", model_id_or_path
            )
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, trust_remote_code=trust_remote_code
        )

        cfg = StreamConfig.from_env(stream_config or StreamConfig(headroom=headroom))
        hw = detect_hardware(cfg.cache_dir)

        # Decide the tier from a cheap meta skeleton BEFORE moving any weights.
        from accelerate import init_empty_weights

        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        graph = discover_graph(skeleton, config)
        max_ctx = max_context or _infer_max_context(config, cfg)
        est = _build_estimate(config, graph, cfg, dtype, quantize, max_ctx)
        decision = select_tier(hw, est, cfg, device=device, tier_override=tier)
        _apply_cache_layers_override(decision, cache_layers)
        _log.info("from_pretrained(%s): %s", model_id_or_path, decision.summary())

        torch_dtype = _resolve_dtype(dtype, config)
        runner: StreamingRunner | None
        if decision.tier == 3:
            from .shard import build_disk_model

            model, runner, graph = build_disk_model(
                model_id_or_path,
                config,
                decision,
                cfg,
                hw,
                shard_path=shard_path,
                dtype=torch_dtype,
                quantize=quantize,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            graph = discover_graph(model, config)
            runner = _attach_runtime(
                model, graph, decision, cfg, hw.resolve_device(device), est, hw
            )

        return cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            graph=graph,
            decision=decision,
            hardware=hw,
            stream_config=cfg,
            estimate=est,
            device=hw.resolve_device(device),
            max_context=max_ctx,
            on_context_overflow=on_context_overflow,
            runner=runner,
        )

    @classmethod
    def from_shards(
        cls,
        shard_path: str | Path,
        *,
        device: str = "auto",
        tier: object = "disk",
        max_context: int | None = None,
        dtype: object | None = None,
        tokenizer: Any = None,
        stream_config: StreamConfig | None = None,
        hardware: HardwareInfo | None = None,
        on_context_overflow: str = "error",
    ) -> StreamModel:
        """Reload a model directly from a streamllm shard directory (prompt §9)."""
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        from .shard import load_manifest, load_sharded_runtime

        out = Path(shard_path)
        config = AutoConfig.from_pretrained(out)
        cfg = StreamConfig.from_env(stream_config or StreamConfig())
        hw = hardware or detect_hardware(cfg.cache_dir)
        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(config)
        graph0 = discover_graph(skeleton, config)
        max_ctx = max_context or _infer_max_context(config, cfg)
        manifest = load_manifest(out)
        quantize = manifest["quantization"]["scheme"]
        est = _build_estimate(config, graph0, cfg, dtype, quantize, max_ctx)
        decision = select_tier(hw, est, cfg, device=device, tier_override=tier)
        model, runner, graph = load_sharded_runtime(
            out, config, decision, cfg, hw, device=decision.compute_device, estimate=est
        )
        return cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            graph=graph,
            decision=decision,
            hardware=hw,
            stream_config=cfg,
            estimate=est,
            device=decision.compute_device,
            max_context=max_ctx,
            on_context_overflow=on_context_overflow,
            runner=runner,
        )

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a forward pass and return logits ``(B, T, vocab)``."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        return out.logits

    # ------------------------------------------------------------- generation

    def generate(
        self,
        inputs: str | list[str] | torch.Tensor,
        *,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        stream: bool = False,
    ) -> str | list[str] | torch.Tensor | Iterator[str]:
        """Generate text/tokens. ``stream=True`` returns an iterator of pieces."""
        params = SamplingParams(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        params.validate()
        input_ids, attention_mask, want_text = self._encode(inputs)
        self._check_context(input_ids.shape[1], max_new_tokens)
        stops = [stop] if isinstance(stop, str) else (list(stop) if stop else [])

        if stream:
            return self._stream_pieces(input_ids, attention_mask, params, max_new_tokens, stops)

        outputs: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
        for b, tok, _piece in self._iter_generate(
            input_ids, attention_mask, params, max_new_tokens, stops
        ):
            outputs[b].append(tok)
        return self._finalize(input_ids, outputs, want_text)

    def _stream_pieces(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        params: SamplingParams,
        max_new_tokens: int,
        stops: list[str],
    ) -> Iterator[str]:
        for _b, _tok, piece in self._iter_generate(
            input_ids, attention_mask, params, max_new_tokens, stops
        ):
            if piece:
                yield piece

    def _iter_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        params: SamplingParams,
        max_new_tokens: int,
        stops: list[str],
    ) -> Iterator[tuple[int, int, str]]:
        from transformers import DynamicCache

        device = self.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch, prompt_len = input_ids.shape

        generator = None
        if params.seed is not None:
            torch.manual_seed(params.seed)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(params.seed)

        metrics = RunMetrics()
        metrics.prompt_tokens = int(attention_mask.sum().item())
        if self.runner is not None:
            self.runner.bind_metrics(metrics)
        metrics.start()

        cache = DynamicCache()
        pos = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
        cache_position = torch.arange(prompt_len, device=device)
        t0 = time.perf_counter()
        logits = self._forward(input_ids, attention_mask, pos, cache, cache_position)[:, -1, :]
        metrics.prefill_s = time.perf_counter() - t0
        metrics.mark_first_token()

        pad_id = self._pad_id()
        stop_ctrl = StopController(
            batch, eos_token_ids=self._eos_ids(), stop_strings=stops, decode=self._decode_ids
        )
        prev: list[list[int]] = [
            input_ids[b][attention_mask[b].bool()].tolist() for b in range(batch)
        ]

        for step in range(max_new_tokens):
            nxt = sample_next(logits, params, prev, generator).to(device)
            for b in range(batch):
                if stop_ctrl.finished[b]:
                    nxt[b, 0] = pad_id
                    continue
                tok = int(nxt[b, 0])
                prev[b].append(tok)
                metrics.generated_tokens += 1
                piece = self._decode_piece(tok)
                yield b, tok, piece
                stop_ctrl.update(b, tok)
            if stop_ctrl.all_done or step == max_new_tokens - 1:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            new_pos = attention_mask.long().sum(-1, keepdim=True) - 1
            past_len = cache.get_seq_length()
            cp = torch.arange(past_len, past_len + 1, device=device)
            t1 = time.perf_counter()
            logits = self._forward(nxt, attention_mask, new_pos, cache, cp)[:, -1, :]
            metrics.decode_s += time.perf_counter() - t1

        metrics.finish()
        if self.runner is not None:
            self.runner.collect_peak(metrics)
        self.last_metrics = metrics

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Any,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
        return out.logits

    # --------------------------------------------------------------- describe

    def describe(self) -> dict[str, Any]:
        """Return detected hardware, chosen tier + deciding numbers, budget (prompt §12)."""
        d = self.decision
        bottleneck = "io-bound" if d.tier == 3 else ("compute-bound" if d.tier == 0 else "mixed")
        resident = ["embeddings", "final_norm", "lm_head"]
        streamed = [] if d.tier == 0 else [f"{self.graph.num_layers} decoder layers"]
        return {
            "model": getattr(self.config, "name_or_path", type(self.config).__name__),
            "tier": d.tier,
            "tier_name": d.name,
            "backing": d.backing,
            "compute_device": d.compute_device,
            "cache_layers": d.cache_layers,
            "reason": d.reason,
            "honest_note": d.honest_note,
            "estimated_bottleneck": bottleneck,
            "resident_components": resident,
            "streamed_components": streamed,
            "hardware": {
                "cuda": self.hardware.cuda_available,
                "devices": [d.name for d in self.hardware.cuda_devices],
                "unified_memory": self.hardware.unified_memory,
                "ram_avail_gb": round(self.hardware.available_ram_bytes / 1e9, 2),
                "disk_free_gb": round(self.hardware.disk_free_bytes / 1e9, 2),
            },
            "budget": self.estimate.as_dict(),
            "deciding_numbers": {
                k: round(v / 1e9, 3)
                for k, v in d.numbers.items()
                if k not in ("headroom", "cache_layers")
            },
            "last_run": self.last_metrics.as_dict() if self.last_metrics else None,
        }

    def shard(self, **kwargs: Any) -> Any:
        """Build disk shards of this model (delegates to :func:`streamllm.shard_model`)."""
        from .shard import shard_model

        return shard_model(self.model, self.config, stream_config=self.stream_config, **kwargs)

    # ------------------------------------------------------------ tokenizer io

    def _encode(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if isinstance(inputs, torch.Tensor):
            ids = inputs if inputs.dim() == 2 else inputs.unsqueeze(0)
            return ids, torch.ones_like(ids), False
        if self.tokenizer is None:
            raise ValueError("string input requires a tokenizer; pass input_ids instead")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        # Left-padding for decoder-only batched generation (prompt §11).
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        return enc["input_ids"], enc["attention_mask"], True

    def _finalize(
        self, input_ids: torch.Tensor, outputs: list[list[int]], want_text: bool
    ) -> str | list[str] | torch.Tensor:
        if want_text:
            texts = [self._decode_ids(o) for o in outputs]
            return texts[0] if len(texts) == 1 else texts
        maxlen = max((len(o) for o in outputs), default=0)
        pad = self._pad_id()
        rows = [o + [pad] * (maxlen - len(o)) for o in outputs]
        return torch.tensor(rows, dtype=torch.long)

    def _decode_ids(self, ids: list[int]) -> str:
        if self.tokenizer is None or not ids:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _decode_piece(self, token_id: int) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    def _eos_ids(self) -> list[int]:
        eos = getattr(self.config, "eos_token_id", None)
        if eos is None and self.tokenizer is not None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            return []
        return [eos] if isinstance(eos, int) else list(eos)

    def _pad_id(self) -> int:
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        pad = getattr(self.config, "pad_token_id", None)
        if pad is not None:
            return int(pad)
        eos = self._eos_ids()
        return eos[0] if eos else 0

    def _check_context(self, prompt_len: int, max_new_tokens: int) -> None:
        limit = int(getattr(self.config, "max_position_embeddings", 0) or 0)
        need = prompt_len + max_new_tokens
        if limit and need > limit:
            if self.on_context_overflow == "truncate":
                _log.warning("context %d > max %d; will be truncated by the model", need, limit)
                return
            raise ContextOverflowError(
                f"prompt_len+max_new_tokens={need} exceeds model max positions {limit}; "
                "raise max_context, shorten the prompt, or set on_context_overflow='truncate'"
            )


class AutoModel:
    """Thin alias so ``AutoModel.from_pretrained`` mirrors the HF entry point."""

    from_pretrained = StreamModel.from_pretrained


def estimate_only(
    model: str | Any,
    *,
    device: str = "auto",
    tier: object = "auto",
    dtype: object | None = None,
    quantize: str | None = None,
    max_context: int | None = None,
    headroom: float = 0.9,
    batch_size: int = 1,
    prompt_len: int | None = None,
    trust_remote_code: bool = False,
    stream_config: StreamConfig | None = None,
    hardware: HardwareInfo | None = None,
) -> dict[str, Any]:
    """Plan a tier from the config alone — **no weights loaded** (prompt §13).

    Powers ``streamllm describe``: works on a machine that cannot run the model.
    It first tries a meta-device skeleton for *exact* per-layer/resident param
    counts (still zero memory, no weights), which makes the estimate accurate for
    any architecture (GPT-2, MoE, ...), and falls back to the analytic SwiGLU
    formula if the skeleton cannot be built (e.g. an architecture the installed
    transformers does not know).
    """
    if isinstance(model, str):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        name = model
    else:
        config = model
        name = getattr(config, "name_or_path", None) or type(config).__name__

    cfg = StreamConfig.from_env(stream_config or StreamConfig(headroom=headroom))
    hw = hardware or detect_hardware(cfg.cache_dir)
    max_ctx = max_context or _infer_max_context(config, cfg)
    pl = prompt_len if prompt_len is not None else (min(max_ctx // 2, 2048) or 1)
    dt = dtype if dtype is not None else _config_dtype_name(config)

    per_layer_params, resident_params = _meta_param_counts(config, trust_remote_code)
    est = estimate_memory(
        config,
        dtype=dt,
        quantize=quantize,
        max_context=max_ctx,
        prompt_len=pl,
        batch_size=batch_size,
        activation_factor=cfg.activation_factor,
        kv_dtype_bytes=cfg.kv_dtype_bytes,
        per_layer_params=per_layer_params,
        resident_params=resident_params,
    )
    decision = select_tier(hw, est, cfg, device=device, tier_override=tier)
    return {
        "model": name,
        "estimate_only": True,
        "tier": decision.tier,
        "tier_name": decision.name,
        "backing": decision.backing,
        "compute_device": decision.compute_device,
        "cache_layers": decision.cache_layers,
        "reason": decision.reason,
        "honest_note": decision.honest_note,
        "is_moe": est.is_moe,
        "num_experts": est.num_experts,
        "hardware": {
            "cuda": hw.cuda_available,
            "devices": [d.name for d in hw.cuda_devices],
            "unified_memory": hw.unified_memory,
            "ram_avail_gb": round(hw.available_ram_bytes / 1e9, 2),
            "disk_free_gb": round(hw.disk_free_bytes / 1e9, 2),
        },
        "budget": est.as_dict(),
        "deciding_numbers": {
            k: round(v / 1e9, 3)
            for k, v in decision.numbers.items()
            if k not in ("headroom", "cache_layers")
        },
    }


# --------------------------------------------------------------------- helpers


def _infer_max_context(config: Any, cfg: StreamConfig) -> int:
    limit = int(getattr(config, "max_position_embeddings", 0) or 0)
    if limit:
        return min(limit, max(cfg.max_context_default, 0) or limit)
    return cfg.max_context_default


def _meta_param_counts(config: Any, trust_remote_code: bool) -> tuple[int | None, int | None]:
    """Exact per-layer + resident param counts from a meta skeleton (no weights).

    Returns ``(None, None)`` if the skeleton cannot be built, so the caller falls
    back to the analytic SwiGLU formula. Building on the meta device allocates no
    tensor storage, so this stays "no weights loaded".
    """
    try:
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        graph = discover_graph(skeleton, config)
        return _measured_param_counts(graph)
    except Exception as exc:
        _log.debug("estimate_only: meta skeleton failed (%s); using analytic counts", exc)
        return None, None


def _build_estimate(
    config: Any,
    graph: ModelGraph,
    cfg: StreamConfig,
    dtype: object | None,
    quantize: str | None,
    max_ctx: int,
) -> MemoryEstimate:
    per_layer, resident = _measured_param_counts(graph)
    # Fall back to analytic counts if the meta modules had no params (defensive).
    if per_layer == 0 or resident == 0:
        counts = count_params_from_config(config)
        per_layer = per_layer or counts.per_layer_params
        resident = resident or counts.resident_params
    dt = dtype if dtype is not None else _config_dtype_name(config)
    prompt_len = min(max_ctx // 2, 2048) or 1
    return estimate_memory(
        config,
        dtype=dt,
        quantize=quantize,
        max_context=max_ctx,
        prompt_len=prompt_len,
        batch_size=1,
        activation_factor=cfg.activation_factor,
        kv_dtype_bytes=cfg.kv_dtype_bytes,
        per_layer_params=per_layer,
        resident_params=resident,
    )


def _config_dtype_name(config: Any) -> str:
    dt = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if dt is None:
        return "bfloat16"
    return str(dt).replace("torch.", "")


def _resolve_dtype(dtype: object | None, config: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype) if dtype is not None else _config_dtype_name(config)
    name = name.replace("torch.", "")
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return table.get(name, torch.float32)


def _apply_cache_layers_override(decision: TierDecision, cache_layers: object) -> None:
    """Honor an explicit ``cache_layers=`` (int) over the auto-computed value."""
    if cache_layers == "auto" or cache_layers is None:
        return
    n = max(int(cache_layers), 1)  # type: ignore[call-overload]
    decision.cache_layers = n
    decision.numbers = {**decision.numbers, "cache_layers": n}
    if n < 2 and decision.tier != 0:
        _log.warning("cache_layers=%d < 2 disables double-buffered prefetch overlap", n)


def _is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error (the demotion trigger, prompt §10)."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "cuda oom" in msg)


def _free_device_memory() -> None:
    import contextlib
    import gc

    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):  # pragma: no cover
            torch.cuda.empty_cache()


def _attach_with_demotion(
    model: torch.nn.Module,
    graph: ModelGraph,
    est: MemoryEstimate,
    cfg: StreamConfig,
    hw: HardwareInfo,
    compute_device: str,
    device_spec: str,
    decision: TierDecision,
    *,
    cache_layers: object,
    allow_demotion: bool,
) -> tuple[TierDecision, StreamingRunner | None]:
    """Attach the runtime, demoting one tier on real OOM (prompt §10).

    Demotion stays within the RAM/GPU tiers (0/1/2): demoting *into* Tier 3 needs
    on-disk shards we cannot synthesize mid-build, so that raises a clear error.
    """
    candidates = [decision.tier]
    if allow_demotion:
        candidates += [t for t in demotion_ladder(decision.tier) if t < 3]

    current = decision
    last_exc: BaseException | None = None
    for i, cand in enumerate(candidates):
        if i > 0:
            prev = candidates[i - 1]
            current = select_tier(hw, est, cfg, device=device_spec, tier_override=cand)
            _apply_cache_layers_override(current, cache_layers)
            _log.warning(
                "graceful demotion: Tier %d OOM'd; retrying at %s — %s",
                prev,
                current.summary(),
                current.reason,
            )
        try:
            runner = _attach_runtime(model, graph, current, cfg, compute_device, est, hw)
            return current, runner
        except Exception as exc:
            if not _is_oom(exc):
                raise
            last_exc = exc
            _log.warning("Tier %d OOM during materialization: %s", cand, exc)
            _free_device_memory()
    raise OutOfMemoryDemotionError(
        f"all attempted tiers {candidates} OOM'd (last: {last_exc}). Lower headroom, "
        "reduce max_context, shard the model and use tier='disk', or pick a smaller model."
    )


def _attach_runtime(
    model: torch.nn.Module,
    graph: ModelGraph,
    decision: TierDecision,
    cfg: StreamConfig,
    compute_device: str,
    est: MemoryEstimate,
    hardware: HardwareInfo,
) -> StreamingRunner | None:
    """Place the model for its tier; install streaming hooks for Tiers 1/2/3.

    Tier 0 fully resides on the compute device. Streaming tiers move resident
    modules to the device and let the :class:`StreamingRunner` swap layer weights
    in/out via forward hooks.
    """
    if decision.tier == 0:
        model.to(compute_device)
        return None

    from .runner import StreamingRunner

    runner = StreamingRunner(
        model, graph, decision, cfg, compute_device, est, unified=hardware.unified_memory
    )
    runner.install()
    return runner
