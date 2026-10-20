# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in V2 reasoning-path suppression; frontend imports stay lightweight."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsContext,
        LogitsProcRequestState,
    )


class ReasoningRepetitionLogitsProcessor(LogitsProcessor):
    """Suppress an observed repeated continuation, not reasoning end tokens.

    Explicitly load this V2 processor and configure
    ``additional_config.reasoning_repetition_guard``. It neither forces a
    transition nor edits generated text. Speculative decoding is rejected until
    draft-prefix marker handling is qualified.
    """

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        import torch

        from vllm.v1.worker.gpu.sample.reasoning_repetition import (
            apply_reasoning_repetition,
        )

        self._apply = apply_reasoning_repetition
        self.req_states = req_states
        additional = vllm_config.additional_config
        if not isinstance(additional, dict):
            raise ValueError("Reasoning repetition guard requires a dictionary")
        config = additional.get("reasoning_repetition_guard", {})
        if not isinstance(config, dict):
            raise ValueError("Reasoning repetition guard requires a dictionary")
        self.ngram_size = config.get("ngram_size", 32)
        self.window_size = config.get("window_size", 512)
        self.max_occurrences = config.get("max_occurrences", 3)
        if (
            any(
                type(v) is not int
                for v in (self.ngram_size, self.window_size, self.max_occurrences)
            )
            or not 2 <= self.ngram_size <= 64
            or not self.ngram_size <= self.window_size <= 2048
            or not 2 <= self.max_occurrences <= self.window_size
        ):
            raise ValueError("Invalid reasoning repetition window configuration")
        reasoning = vllm_config.reasoning_config
        start = reasoning.reasoning_start_token_ids if reasoning else None
        end = (
            reasoning.natural_reasoning_end_token_ids
            or reasoning.reasoning_end_token_ids
            if reasoning
            else None
        )
        if not start or not end:
            raise ValueError("Reasoning repetition guard requires reasoning token IDs")
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "V2 reasoning repetition guard requires non-speculative decode"
            )
        if vllm_config.structured_outputs_config.enable_in_reasoning:
            raise ValueError("Cannot constrain structured reasoning")
        device = req_states.device
        self.start = torch.tensor(start, dtype=torch.int32, device=device)
        self.end = torch.tensor(end, dtype=torch.int32, device=device)
        self.enabled = torch.zeros(
            req_states.max_num_reqs, dtype=torch.int32, device=device
        )
        self.last_start = torch.full_like(self.enabled, -1)
        self.last_end = torch.full_like(self.enabled, -1)
        self.scan_pos = torch.zeros_like(self.enabled)
        self.protected = torch.zeros(
            (req_states.max_num_reqs, req_states.vocab_size),
            dtype=torch.bool,
            device=device,
        )
        self.pending: dict[int, set[int]] = {}

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.pending[req_idx] = set(sampling_params.all_stop_token_ids)
        return True

    def apply_staged_writes(self) -> None:
        for index, stops in self.pending.items():
            self.enabled[index] = 0
            self.last_start[index] = -1
            self.last_end[index] = -1
            self.scan_pos[index] = 0
            self.protected[index].fill_(False)
            if stops:
                self.protected[index, list(stops)] = True
            self.protected[index, self.start.long()] = True
            self.protected[index, self.end.long()] = True
        self.pending.clear()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        self._apply(
            logits,
            ctx,
            self.req_states,
            self.enabled,
            self.last_start,
            self.last_end,
            self.scan_pos,
            self.start,
            self.end,
            self.protected,
            self.ngram_size,
            self.window_size,
            self.max_occurrences,
        )
        return logits
