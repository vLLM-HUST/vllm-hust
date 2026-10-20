# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-only bounded-window matching for the opt-in V2 reasoning guard."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.thinking_budget import (
    _COLD_SCAN_BLOCK,
    _update_committed_marker_cache_kernel,
)


@triton.jit
def _repeat_kernel(
    logits,
    logits_stride,
    mapping,
    tokens,
    token_stride,
    prompt_lens,
    total_lens,
    starts,
    ends,
    protected,
    candidates,
    NGRAM: tl.constexpr,
    WINDOW: tl.constexpr,
    OCCURRENCES: tl.constexpr,
    VOCAB: tl.constexpr,
    START_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tl.store(candidates + row, -1)
    slot = tl.load(mapping + row)
    start = tl.load(starts + slot)
    end = tl.load(ends + slot)
    length = tl.load(total_lens + slot)
    if start < 0 or start <= end:
        return
    # Prompt text establishes the reasoning mode but never counts as repetition.
    lower = tl.maximum(tl.load(prompt_lens + slot), start + START_LEN)
    lower = tl.maximum(lower, length - WINDOW)
    positions = lower + tl.arange(0, BLOCK)
    valid = positions + NGRAM <= length
    match = valid
    for j in tl.static_range(NGRAM - 1):
        actual = tl.load(
            tokens + slot * token_stride + positions + j, mask=valid, other=-1
        )
        suffix = tl.load(
            tokens + slot * token_stride + length - NGRAM + 1 + j,
            mask=length - NGRAM + 1 >= lower,
            other=-2,
        )
        match = match & (actual == suffix)
    latest = tl.max(tl.where(match, positions, -1), 0)
    if latest < 0:
        return
    candidate = tl.load(tokens + slot * token_stride + latest + NGRAM - 1)
    # Repeated phrasing in distinct paragraphs is not a closed generation loop.
    # Require an uninterrupted periodic suffix, not just distant n-gram matches.
    period = length - NGRAM + 1 - latest
    # Overlapping n-grams still establish distinct short-period continuations;
    # do not wait NGRAM * OCCURRENCES tokens for a two-token loop.
    required = tl.maximum(NGRAM + period * (OCCURRENCES - 1), period * OCCURRENCES)
    if length - required < lower:
        return
    offsets = tl.arange(0, BLOCK)
    within = offsets < required - period
    recent_pos = length - 1 - offsets
    recent = tl.load(tokens + slot * token_stride + recent_pos, mask=within, other=-1)
    previous = tl.load(
        tokens + slot * token_stride + recent_pos - period, mask=within, other=-1
    )
    if tl.sum((within & (recent != previous)).to(tl.int32), 0) > 0:
        return
    next_ids = tl.load(
        tokens + slot * token_stride + positions + NGRAM - 1, mask=valid, other=-1
    )
    count = tl.sum((match & (next_ids == candidate)).to(tl.int32), 0)
    safe = ~tl.load(protected + slot * VOCAB + candidate)
    if count < OCCURRENCES or not safe:
        return
    candidate_logit = tl.load(logits + row * logits_stride + candidate)
    if not tl.abs(candidate_logit) < float("inf"):
        return
    tl.store(candidates + row, candidate)


@triton.jit
def _mask_repeat_kernel(
    logits, logits_stride, candidates, VOCAB: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    candidate = tl.load(candidates + row)
    if candidate < 0:
        return
    ids = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        logits + row * logits_stride + ids, mask=ids < VOCAB, other=-float("inf")
    )
    alternatives = (ids != candidate) & (tl.abs(values) < float("inf"))
    if tl.sum(alternatives.to(tl.int32), 0) > 0:
        # Other finite tokens never change; concurrent tiles may safely mask the
        # same candidate, but cannot remove a row's sole valid continuation.
        tl.atomic_min(
            logits + row * logits_stride + candidate, -float("inf"), sem="relaxed"
        )


def apply_reasoning_repetition(
    logits: torch.Tensor,
    ctx: LogitsContext,
    states: LogitsProcRequestState,
    enabled: torch.Tensor,
    last_start: torch.Tensor,
    last_end: torch.Tensor,
    scan_pos: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    protected: torch.Tensor,
    ngram_size: int,
    window_size: int,
    max_occurrences: int,
) -> None:
    # Reuse the qualified incremental marker scan, not its budget-forcing stage.
    _update_committed_marker_cache_kernel[(ctx.idx_mapping.shape[0],)](
        ctx.idx_mapping,
        enabled,
        states.all_token_ids.gpu,
        states.all_token_ids.gpu.stride(0),
        states.total_len.gpu,
        last_start,
        last_end,
        scan_pos,
        start,
        end,
        START_LEN=start.numel(),
        NATURAL_END_LEN=end.numel(),
        MAX_LEN=max(start.numel(), end.numel()),
        BLOCK=_COLD_SCAN_BLOCK,
    )
    candidates = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    _repeat_kernel[(logits.shape[0],)](
        logits,
        logits.stride(0),
        ctx.expanded_idx_mapping,
        states.all_token_ids.gpu,
        states.all_token_ids.gpu.stride(0),
        states.prompt_len.gpu,
        states.total_len.gpu,
        last_start,
        last_end,
        protected,
        candidates,
        NGRAM=ngram_size,
        WINDOW=window_size,
        OCCURRENCES=max_occurrences,
        VOCAB=logits.shape[1],
        START_LEN=start.numel(),
        BLOCK=triton.next_power_of_2(window_size),
    )
    _mask_repeat_kernel[(logits.shape[0], triton.cdiv(logits.shape[1], 1024))](
        logits, logits.stride(0), candidates, VOCAB=logits.shape[1], BLOCK=1024
    )
