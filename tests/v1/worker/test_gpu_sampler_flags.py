# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


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
            [90] + list(range(10, 42)) * 4 + [50] * 600 + list(range(10, 41)),
            1,
            None,
            32,
        ),
    ],
)
def test_v2_reasoning_guard_only_masks_observed_reasoning_loops(
    tokens, prompt_len, expected_blocked, ngram_size
):
    """Keep prompt repetitions and all post-thinking logits byte-identical."""
    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsContext,
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
        additional_config={"reasoning_repetition_guard": {"ngram_size": ngram_size}},
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
    sampler.add_request(3, prompt_len, SamplingParams(temperature=0))
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


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
    )


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(), False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, id="penalty"),
        pytest.param(SamplingParams(_bad_words_token_ids=[[1]]), True, id="bad-words"),
        pytest.param(SamplingParams(temperature=0.7), True, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(), True, id="all-logits-processors"
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams, expected: bool
):
    sampler = _make_sampler()
    sampler.add_request(3, prompt_len=1, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, 1, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, 1, SamplingParams())

    assert not sampler.needs_logits_processing[3]


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(2, 1, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])
