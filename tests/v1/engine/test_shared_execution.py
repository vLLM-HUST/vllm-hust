# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored
from vllm.v1.shared_execution import (
    SharedExecutionMetadata,
    analyze_approximate_kv_sharing,
    compare_kv_event_streams,
    evaluate_exact_kv_reuse,
    extract_shared_execution_metadata,
    summarize_kv_cache_events,
)


pytestmark = pytest.mark.skip_global_cleanup


def test_extract_shared_execution_metadata_from_sampling_params():
    sampling_params = SamplingParams(
        max_tokens=16,
        extra_args={
            "shared_execution": {
                "tenant": "tenant-a",
                "canonical_prefix_key": "prefix-123",
                "retrieval_keys": ["doc-1", "doc-2"],
                "tool_calls": ["search", "search", "rerank"],
                "priority_hint": 7,
            }
        },
    )

    metadata = extract_shared_execution_metadata(sampling_params, None)

    assert metadata == SharedExecutionMetadata(
        tenant="tenant-a",
        canonical_prefix_key="prefix-123",
        retrieval_signature=("doc-1", "doc-2"),
        tool_signature=("search", "search", "rerank"),
        priority_hint=7,
    )


def test_extract_shared_execution_metadata_from_pooling_params():
    pooling_params = PoolingParams(
        extra_kwargs={
            "shared_execution": {
                "tenant": "tenant-b",
                "prefix_key": "prefix-xyz",
                "stage_id": "retrieval",
            }
        }
    )

    metadata = extract_shared_execution_metadata(None, pooling_params)

    assert metadata == SharedExecutionMetadata(
        tenant="tenant-b",
        canonical_prefix_key="prefix-xyz",
        stage_id="retrieval",
    )


def test_evaluate_exact_kv_reuse_requires_tenant_and_stage_compatibility():
    anchor = SharedExecutionMetadata(
        tenant="tenant-a",
        canonical_prefix_key="prefix-123",
        stage_id="retrieval",
    )

    candidate = SharedExecutionMetadata(
        tenant="tenant-b",
        canonical_prefix_key="prefix-123",
        stage_id="retrieval",
    )
    decision = evaluate_exact_kv_reuse(anchor, candidate)
    assert not decision.can_reuse
    assert decision.reason == "tenant_mismatch"

    candidate = SharedExecutionMetadata(
        tenant="tenant-a",
        canonical_prefix_key="prefix-123",
        stage_id="tool",
    )
    decision = evaluate_exact_kv_reuse(anchor, candidate)
    assert not decision.can_reuse
    assert decision.reason == "stage_mismatch"

    candidate = SharedExecutionMetadata(
        tenant="tenant-a",
        canonical_prefix_key="prefix-123",
        stage_id="retrieval",
    )
    decision = evaluate_exact_kv_reuse(anchor, candidate)
    assert decision.can_reuse
    assert decision.reason == "canonical_prefix_match"


def test_analyze_approximate_kv_sharing_uses_signature_overlap():
    anchor = SharedExecutionMetadata(
        tenant="tenant-a",
        stage_id="retrieval",
        retrieval_signature=("doc-1", "doc-2"),
        tool_signature=("search", "rerank"),
    )
    candidate = SharedExecutionMetadata(
        tenant="tenant-a",
        stage_id="retrieval",
        retrieval_signature=("doc-2", "doc-3"),
        tool_signature=("search",),
    )

    analysis = analyze_approximate_kv_sharing(anchor, candidate)

    assert analysis.label == "partial"
    assert analysis.reason == "signature_overlap"
    assert analysis.tenant_compatible
    assert analysis.stage_compatible
    assert analysis.retrieval_overlap > 0
    assert analysis.tool_overlap > 0


def test_analyze_approximate_kv_sharing_caps_cross_tenant_score():
    anchor = SharedExecutionMetadata(
        tenant="tenant-a",
        stage_id="retrieval",
        retrieval_signature=("doc-1",),
        tool_signature=("search",),
    )
    candidate = SharedExecutionMetadata(
        tenant="tenant-b",
        stage_id="retrieval",
        retrieval_signature=("doc-1",),
        tool_signature=("search",),
    )

    analysis = analyze_approximate_kv_sharing(anchor, candidate)

    assert not analysis.tenant_compatible
    assert analysis.score < 0.5
    assert analysis.reason == "tenant_mismatch"


def test_compare_kv_event_streams_reports_mismatched_blocks():
    reference_events = [
        BlockStored(
            block_hashes=[101, 102],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4],
            block_size=2,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        ),
        BlockRemoved(block_hashes=[201], medium="GPU"),
        AllBlocksCleared(),
    ]
    candidate_events = [
        BlockStored(
            block_hashes=[101],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        ),
        BlockRemoved(block_hashes=[202], medium="GPU"),
    ]

    summary = summarize_kv_cache_events(reference_events)
    assert summary.stored_hashes == (101, 102)
    assert summary.removed_hashes == (201,)
    assert summary.clear_event_count == 1

    report = compare_kv_event_streams(reference_events, candidate_events)

    assert not report.is_consistent
    assert report.issues == (
        "clear_event_count_mismatch",
        "stored_hash_mismatch",
        "removed_hash_mismatch",
    )
    assert report.missing_stores == (102,)
    assert report.extra_removes == (202,)


def test_compare_kv_event_streams_accepts_equivalent_batches():
    reference_events = [
        BlockStored(
            block_hashes=[301],
            parent_block_hash=None,
            token_ids=[7, 8],
            block_size=2,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        ),
        BlockRemoved(block_hashes=[401], medium="GPU"),
    ]
    candidate_events = [
        BlockStored(
            block_hashes=[301],
            parent_block_hash=None,
            token_ids=[7, 8],
            block_size=2,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        ),
        BlockRemoved(block_hashes=[401], medium="GPU"),
    ]

    report = compare_kv_event_streams(reference_events, candidate_events)

    assert report.is_consistent
    assert report.issues == ()