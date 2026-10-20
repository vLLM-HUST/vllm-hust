# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 logits processor tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
TARGET_TOKEN = 7
BIAS = 5.0
BOOST = 100.0
TEMPERATURE = 2.0


@pytest.mark.parametrize("max_occurrences", [3, 4])
@pytest.mark.parametrize(
    "tokens,prompt_len,expected_blocked,ngram_size",
    [
        ([90] + [10, 11] * 4 + [10], 1, 11, 2),
        ([90] + [10, 11] * 2 + [10], 1, None, 2),
        ([90] + [10, 11] * 4 + [10, 91, 10], 1, None, 2),
        ([90] + [10, 11] * 4 + [10], 9, None, 2),
        ([1] + [10, 11] * 4 + [10], 1, None, 2),
        ([90] + [10, 91] * 4 + [10], 1, None, 2),
        ([90] + list(range(10, 42)) * 4 + list(range(10, 41)), 1, 41, 32),
        (
            [90]
            + sum([list(range(10, 42)) + [50 + i] for i in range(4)], [])
            + list(range(10, 41)),
            1,
            None,
            32,
        ),
        ([90] + [10] * 128, 1, 10, 32),
        (
            [90] + list(range(10, 42)) * 4 + [50] * 600 + list(range(10, 41)),
            1,
            None,
            32,
        ),
    ],
)
def test_v2_reasoning_guard_only_masks_observed_reasoning_loops(
    tokens, prompt_len, expected_blocked, ngram_size, max_occurrences
):
    """Keep prompt repetitions and all post-thinking logits byte-identical."""
    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsProcRequestState,
    )
    from vllm.v1.worker.gpu.sample.logits_processor.reasoning_repetition import (
        ReasoningRepetitionLogitsProcessor,
    )

    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 1),
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="guard",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    config = SimpleNamespace(
        additional_config={
            "reasoning_repetition_guard": {
                "ngram_size": ngram_size,
                "max_occurrences": max_occurrences,
            }
        },
        reasoning_config=MockReasoningConfig(),
        speculative_config=None,
        structured_outputs_config=SimpleNamespace(enable_in_reasoning=False),
    )
    proc = ReasoningRepetitionLogitsProcessor(
        config, LogitsProcRequestState.from_request_state(req_states)
    )
    proc.add_request(3, SamplingParams())
    proc.apply_staged_writes()
    mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    ctx = LogitsContext(
        expanded_idx_mapping=mapping,
        idx_mapping=mapping,
        idx_mapping_np=np.array([3]),
        expanded_local_pos=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        input_ids=torch.tensor([tokens[-1]], dtype=torch.int32, device=DEVICE),
        pos=torch.tensor([len(tokens) - 1], dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.array([len(tokens)]),
    )
    logits = torch.arange(VOCAB_SIZE, dtype=torch.float32, device=DEVICE)[None]
    expected = logits.clone()
    if expected_blocked is not None:
        expected[0, expected_blocked] = -float("inf")
    torch.testing.assert_close(proc.apply(logits, ctx), expected)
    # Native configuration must reach the actual sampler, including greedy mode.
    sampler = Sampler(
        vllm_config=config,
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
    )
    sampler.add_request(3, SamplingParams(temperature=0))
    sampler.apply_staged_writes()
    assert sampler.needs_logits_processing[3]
    native_logits = torch.arange(VOCAB_SIZE, dtype=torch.float32, device=DEVICE)[None]
    native_out = sampler.apply_sampling_params(
        native_logits,
        mapping,
        mapping,
        ctx.idx_mapping_np,
        ctx.pos,
        ctx.input_ids,
        ctx.expanded_local_pos,
        ctx.seq_lens_upper_bound_np,
    )
    torch.testing.assert_close(native_out, expected)
    # A grammar row with only one valid continuation must not become empty.
    single = torch.full_like(logits, -float("inf"))
    single[0, 11] = 0
    expected_single = single.clone()
    torch.testing.assert_close(proc.apply(single, ctx), expected_single)
    # Explicit stop IDs, including a repeated continuation, are protected.
    proc.add_request(3, SamplingParams(stop_token_ids=[expected_blocked or 11]))
    proc.apply_staged_writes()
    untouched = torch.zeros_like(logits)
    torch.testing.assert_close(proc.apply(untouched, ctx), torch.zeros_like(logits))


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


class RecordingProcessor(LogitsProcessor):
    """Adds BOOST to TARGET_TOKEN and records every lifecycle callback."""

    def __init__(self, vllm_config: Any, req_state: Any):
        self.events: list[str] = []
        self.seen_logits: torch.Tensor | None = None

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.events.append(f"add_request:{req_idx}")
        return True

    def apply_staged_writes(self) -> None:
        self.events.append("apply_staged_writes")

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        self.events.append("apply")
        self.seen_logits = logits.clone()
        logits[:, TARGET_TOKEN] += BOOST
        return logits


def _make_sampler(processor: RecordingProcessor) -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=[processor],
    )


def test_custom_processor_lifecycle_and_pipeline_position():
    """A custom processor's hooks fire in order, it runs after the built-in
    bias stage, and before temperature scaling."""
    proc = RecordingProcessor(None, None)
    sampler = _make_sampler(proc)
    sampler.add_request(
        0,
        SamplingParams(logit_bias={TARGET_TOKEN: BIAS}, temperature=TEMPERATURE),
    )
    sampler.apply_staged_writes()

    num_reqs = 1
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    raw = torch.zeros(num_reqs, VOCAB_SIZE, device=DEVICE)
    out = sampler.apply_sampling_params(
        raw,
        expanded_idx_mapping=idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.ones(num_reqs, dtype=np.int64),
    )

    assert proc.events == [
        "add_request:0",
        "apply_staged_writes",
        "apply",
    ]
    # The processor sees the built-in bias applied but no temperature scaling.
    assert proc.seen_logits is not None
    assert proc.seen_logits[0, TARGET_TOKEN].item() == BIAS
    # Its own modification is then temperature-scaled by the sampler.
    assert out[0, TARGET_TOKEN].item() == (BIAS + BOOST) / TEMPERATURE
