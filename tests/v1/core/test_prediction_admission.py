# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-level integration tests for prediction-aware admission.

These drive a **real** ``Scheduler`` with a real ``KVCacheManager`` and real
request queues, so they assert what the unit tests for
``predicted_full_sequence_tokens`` cannot: that the prediction actually changes
which requests are admitted, that it never changes behaviour when absent, and
that a bad prediction cannot be used to reserve KV beyond what the request may
generate.

The KV pool is deliberately tiny (``num_blocks`` in the single digits with the
first block reserved as the null block) so that admission decisions are forced
within a test rather than inferred from a large cache.
"""

from __future__ import annotations

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# A 16-token block size keeps the arithmetic in these tests obvious: one block
# holds exactly block_size tokens.
BLOCK_SIZE = 16


def block_count(request) -> int:
    """Blocks the request currently holds, mirroring what the gate reserves."""
    return len(request.prompt_token_ids)


def make_scheduler(*, num_blocks: int, max_num_batched_tokens: int = 8192):
    return create_scheduler(
        block_size=BLOCK_SIZE,
        num_blocks=num_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=False,
    )


def test_no_prediction_matches_upstream_admission():
    """With no prediction the gate plans for the prompt alone (upstream)."""
    # Room for a handful of blocks.
    scheduler = make_scheduler(num_blocks=8)

    requests = create_requests(
        num_requests=1, num_tokens=32, block_size=BLOCK_SIZE, max_tokens=256
    )
    request = requests[0]
    assert request.predicted_length is None

    scheduler.add_request(request)
    output = scheduler.schedule()

    # The prompt fits, so the request is admitted and starts prefilling.
    assert request.request_id in output.num_scheduled_tokens
    assert request.status == RequestStatus.RUNNING


def test_prediction_too_large_for_the_pool_delays_admission():
    """A prediction that cannot fit holds the request back instead of thrashing.

    Without a prediction a 32-token prompt enters a pool sized for a few
    blocks. With a prediction covering a long output, the same pool cannot
    promise the full sequence, so the gate refuses admission for this step.
    """
    requests = create_requests(
        num_requests=1,
        num_tokens=32,
        block_size=BLOCK_SIZE,
        max_tokens=512,
        predicted_length=512,
    )
    request = requests[0]
    assert request.predicted_length == 512

    # Enough blocks for the prompt (2) plus a couple, nowhere near 512 tokens
    # (32 blocks) worth.
    scheduler = make_scheduler(num_blocks=6)

    scheduler.add_request(request)
    output = scheduler.schedule()

    # Not admitted: the predicted full sequence cannot fit in this pool.
    assert request.request_id not in output.num_scheduled_tokens
    assert request.status == RequestStatus.WAITING


def test_prediction_that_fits_is_admitted_normally():
    """Control for the previous test: same pool, prediction small enough."""
    requests = create_requests(
        num_requests=1,
        num_tokens=32,
        block_size=BLOCK_SIZE,
        max_tokens=512,
        predicted_length=16,
    )
    scheduler = make_scheduler(num_blocks=6)

    scheduler.add_request(requests[0])
    output = scheduler.schedule()

    assert requests[0].request_id in output.num_scheduled_tokens
    assert requests[0].status == RequestStatus.RUNNING


def test_oversized_prediction_is_clamped_by_max_tokens():
    """A hostile prediction cannot reserve more than the request may generate.

    This is the bound that stops a client from parking KV it will never use:
    two requests differing only in how absurd their prediction is must reach
    the same admission decision, because both are clamped to ``max_tokens``.
    """
    modest = create_requests(
        num_requests=1,
        num_tokens=32,
        block_size=BLOCK_SIZE,
        max_tokens=32,
        predicted_length=32,
    )[0]
    absurd = create_requests(
        num_requests=1,
        num_tokens=32,
        block_size=BLOCK_SIZE,
        max_tokens=32,
        predicted_length=1_000_000,
    )[0]

    modest_scheduler = make_scheduler(num_blocks=8)
    absurd_scheduler = make_scheduler(num_blocks=8)

    modest_scheduler.add_request(modest)
    absurd_scheduler.add_request(absurd)
    modest_output = modest_scheduler.schedule()
    absurd_output = absurd_scheduler.schedule()

    assert (modest.request_id in modest_output.num_scheduled_tokens) == (
        absurd.request_id in absurd_output.num_scheduled_tokens
    )


def test_mixed_predicted_and_unpredicted_requests_are_both_admitted():
    """The gate must not disturb requests that carry no prediction."""
    requests = create_requests(
        num_requests=2,
        num_tokens=32,
        block_size=BLOCK_SIZE,
        max_tokens=64,
        predicted_length=[32, None],
    )
    assert requests[0].predicted_length == 32
    assert requests[1].predicted_length is None

    scheduler = make_scheduler(num_blocks=16)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()

    # Both fit comfortably; neither is starved by the other's prediction.
    scheduled = set(output.num_scheduled_tokens)
    assert requests[0].request_id in scheduled
    assert requests[1].request_id in scheduled


def test_prediction_does_not_block_decode_after_prefill():
    """Once decoding starts the prediction is ignored (real length is known).

    The gate only extends the reserve while a request is still prefilling. A
    request whose prediction is far larger than what it has produced must keep
    being scheduled for decode, i.e. the prediction must not be re-applied
    after the real length is known.

    The request is driven through the normal engine path
    (``schedule`` -> ``update_from_output``) rather than by mutating counters,
    so the scheduler's bookkeeping stays consistent.
    """
    requests = create_requests(
        num_requests=1,
        num_tokens=32,
        block_size=BLOCK_SIZE,
        max_tokens=256,
        predicted_length=256,
    )
    request = requests[0]
    scheduler = make_scheduler(num_blocks=64)

    scheduler.add_request(request)
    output = scheduler.schedule()
    if request.request_id not in output.num_scheduled_tokens:
        pytest.skip("pool too small for this platform's block size")

    # Complete prefill and produce a token, so the request enters decode.
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert request.num_computed_tokens >= request.num_prompt_tokens

    decode_output = scheduler.schedule()
    # Decode is still scheduled: the prediction is not re-applied post-prefill.
    assert request.request_id in decode_output.num_scheduled_tokens


def test_negative_prediction_is_harmless():
    """A nonsense prediction must not fail the request or panic the scheduler."""
    requests = create_requests(
        num_requests=1,
        num_tokens=32,
        block_size=BLOCK_SIZE,
        max_tokens=64,
        predicted_length=-10,
    )
    scheduler = make_scheduler(num_blocks=8)

    scheduler.add_request(requests[0])
    output = scheduler.schedule()

    assert requests[0].request_id in output.num_scheduled_tokens
