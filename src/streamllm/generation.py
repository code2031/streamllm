"""Generation: logits processing, sampling, stopping, streaming (prompt §11).

Pure, tier-agnostic helpers. The decode loop in :mod:`streamllm.model` calls
:func:`sample_next` and :class:`StopController`; whether weights stream or not is
invisible here. Logits-processing order matches HF: repetition penalty →
temperature → top-k → top-p → min-p → softmax → sample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

# A streaming callback receives (batch_index, token_id, decoded_text_piece).
StreamCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class SamplingParams:
    """Decoding controls (prompt §11)."""

    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    seed: int | None = None

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0 or None")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1] or None")
        if self.min_p is not None and not (0 <= self.min_p <= 1):
            raise ValueError("min_p must be in [0, 1] or None")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")


def _apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: list[list[int]], penalty: float
) -> torch.Tensor:
    if penalty == 1.0 or not any(prev_tokens):
        return logits
    for b, toks in enumerate(prev_tokens):
        if not toks:
            continue
        idx = torch.tensor(sorted(set(toks)), device=logits.device, dtype=torch.long)
        scores = logits[b, idx]
        logits[b, idx] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits


def _top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def _top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    # Remove tokens once cumulative prob exceeds p, always keeping the top token.
    remove = cum - probs > p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)


def _min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    top = probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < min_p * top, float("-inf"))


def sample_next(
    logits: torch.Tensor,
    params: SamplingParams,
    prev_tokens: list[list[int]],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return the next token id per row, shape ``(B, 1)``.

    ``logits`` is ``(B, vocab)`` (last position). ``prev_tokens`` is the
    per-row history used for repetition penalty.
    """
    logits = logits.float()
    logits = _apply_repetition_penalty(logits, prev_tokens, params.repetition_penalty)

    if not params.do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    if params.temperature != 1.0:
        logits = logits / max(params.temperature, 1e-6)
    if params.top_k is not None:
        logits = _top_k(logits, params.top_k)
    if params.top_p is not None:
        logits = _top_p(logits, params.top_p)
    if params.min_p is not None:
        logits = _min_p(logits, params.min_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


class StopController:
    """Tracks per-sequence completion: EOS, max_new_tokens, and stop strings.

    Stop strings are decode-aware: we keep a small text tail per row and check
    for any stop substring after each token. A finished row is masked: its token
    stops updating and generation ends when *all* rows finish (prompt §11).
    """

    def __init__(
        self,
        batch_size: int,
        *,
        eos_token_ids: Sequence[int] | None,
        stop_strings: Sequence[str] | None,
        decode: Callable[[list[int]], str],
    ) -> None:
        self.batch = batch_size
        self.eos = set(eos_token_ids or [])
        self.stops = list(stop_strings or [])
        self._decode = decode
        self.finished = [False] * batch_size
        self._tails: list[list[int]] = [[] for _ in range(batch_size)]
        # Longest stop string in tokens is unknown; keep a generous text tail.
        self._tail_keep = max((len(s) for s in self.stops), default=0) + 8

    def update(self, b: int, token_id: int) -> bool:
        """Register ``token_id`` for row ``b``; return True if the row is now done."""
        if self.finished[b]:
            return True
        if token_id in self.eos:
            self.finished[b] = True
            return True
        if self.stops:
            self._tails[b].append(token_id)
            text = self._decode(self._tails[b][-self._tail_keep :])
            if any(s in text for s in self.stops):
                self.finished[b] = True
                return True
        return False

    @property
    def all_done(self) -> bool:
        return all(self.finished)
