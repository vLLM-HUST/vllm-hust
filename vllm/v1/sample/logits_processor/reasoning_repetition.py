# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, bounded-window n-gram suppression confined to reasoning sections."""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class _ReasoningRepetitionState:
    output: list[int]
    protected: set[int]
    processed: int = 0
    interventions: int = 0
    active: bool = False
    marker_tail: tuple[int, ...] = ()
    tail: deque[int] = field(default_factory=deque)
    history: deque[tuple[int, ...]] = field(default_factory=deque)
    counts: dict[tuple[int, ...], Counter[int]] = field(default_factory=dict)

    def reset_history(self) -> None:
        self.tail.clear()
        self.history.clear()
        self.counts.clear()


class ReasoningRepetitionLogitsProcessor(LogitsProcessor):
    """Block repeated n-gram continuations, never force a reasoning transition.

    Load explicitly with ``--logits-processors`` and configure
    ``additional_config.reasoning_repetition_guard``. No default sampling policy
    or final-answer logits are changed. Speculative decoding is unsupported by
    the custom-logits-processor interface.
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        additional = vllm_config.additional_config
        if not isinstance(additional, dict):
            raise ValueError(
                "Reasoning repetition guard requires dictionary configuration"
            )
        config = additional.get("reasoning_repetition_guard", {})
        if not isinstance(config, dict):
            raise ValueError(
                "Reasoning repetition guard requires dictionary configuration"
            )
        self.ngram_size = config.get("ngram_size", 32)
        self.max_occurrences = config.get("max_occurrences", 3)
        self.window_size = config.get("window_size", 512)
        if (
            any(
                type(value) is not int
                for value in (
                    self.ngram_size,
                    self.max_occurrences,
                    self.window_size,
                )
            )
            or self.ngram_size < 2
            or self.max_occurrences < 2
            or self.window_size < self.ngram_size
        ):
            raise ValueError("Invalid reasoning repetition window configuration")
        reasoning = vllm_config.reasoning_config
        self.start = (
            tuple(reasoning.reasoning_start_token_ids or ()) if reasoning else ()
        )
        self.end = (
            tuple(
                getattr(reasoning, "natural_reasoning_end_token_ids", None)
                or reasoning.reasoning_end_token_ids
                or ()
            )
            if reasoning
            else ()
        )
        if not self.start or not self.end:
            raise ValueError("Reasoning repetition guard requires reasoning token IDs")
        if vllm_config.structured_outputs_config.enable_in_reasoning:
            raise ValueError(
                "Reasoning repetition guard cannot constrain structured reasoning"
            )
        self.marker_size = max(len(self.start), len(self.end))
        self.states: dict[int, _ReasoningRepetitionState] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def _accept(
        self, state: _ReasoningRepetitionState, token: int, *, record: bool = True
    ) -> None:
        state.marker_tail = (*state.marker_tail, token)[-self.marker_size :]
        if state.marker_tail[-len(self.start) :] == self.start:
            state.active = True
            state.reset_history()
            return
        if state.marker_tail[-len(self.end) :] == self.end:
            state.active = False
            state.reset_history()
            return
        if not state.active or not record:
            return
        state.tail.append(token)
        if len(state.tail) < self.ngram_size:
            return
        ngram = tuple(state.tail)
        state.counts.setdefault(ngram[:-1], Counter())[ngram[-1]] += 1
        state.history.append(ngram)
        state.tail.popleft()
        if len(state.history) > self.window_size - self.ngram_size + 1:
            expired = state.history.popleft()
            counts = state.counts[expired[:-1]]
            counts[expired[-1]] -= 1
            if not counts[expired[-1]]:
                del counts[expired[-1]]
            if not counts:
                del state.counts[expired[:-1]]

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update:
            for index in batch_update.removed:
                self.states.pop(index, None)
            for index, params, prompt, output in batch_update.added:
                state = _ReasoningRepetitionState(
                    output, set(params.all_stop_token_ids) | set(self.start + self.end)
                )
                for token in prompt or ():
                    self._accept(state, token, record=False)
                self.states[index] = state
            for source, destination, direction in batch_update.moved:
                first = self.states.pop(source, None)
                second = self.states.pop(destination, None)
                if first is not None:
                    self.states[destination] = first
                if second is not None and direction == MoveDirectionality.SWAP:
                    self.states[source] = second
        for state in self.states.values():
            while state.processed < len(state.output):
                token = state.output[state.processed]
                if token < 0:
                    break
                self._accept(state, token)
                state.processed += 1

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, state in self.states.items():
            if not state.active:
                continue
            counts = state.counts.get(tuple(state.tail))
            if not counts:
                continue
            blocked = [
                token
                for token, count in counts.items()
                if count >= self.max_occurrences and token not in state.protected
            ]
            if blocked:
                if not state.interventions:
                    logger.info(
                        "Detected repeated reasoning path at output token %d "
                        "(ngram_size=%d, max_occurrences=%d)",
                        state.processed,
                        self.ngram_size,
                        self.max_occurrences,
                    )
                state.interventions += 1
                previous = logits[index, blocked].clone()
                logits[index, blocked] = -float("inf")
                # Do not invalidate a row already constrained by other processors.
                logits[index, blocked] = torch.where(
                    torch.isfinite(logits[index]).any(),
                    -float("inf"),
                    previous,
                )
        return logits
