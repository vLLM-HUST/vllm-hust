# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import msgspec

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored


class SharedExecutionMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    tenant: str | None = None
    canonical_prefix_key: str | None = None
    retrieval_signature: tuple[str, ...] = ()
    tool_signature: tuple[str, ...] = ()
    stage_id: str | None = None
    priority_hint: int | None = None

    def has_canonical_prefix(self) -> bool:
        return bool(self.canonical_prefix_key)

    def shares_canonical_prefix_with(
        self,
        other: "SharedExecutionMetadata | None",
    ) -> bool:
        return (
            other is not None
            and self.canonical_prefix_key is not None
            and self.canonical_prefix_key == other.canonical_prefix_key
        )

    def is_exact_kv_reuse_candidate(
        self,
        other: "SharedExecutionMetadata | None",
    ) -> bool:
        return evaluate_exact_kv_reuse(self, other).can_reuse


@dataclass(frozen=True)
class ExactKVReuseDecision:
    can_reuse: bool
    reason: str
    tenant_compatible: bool
    stage_compatible: bool
    canonical_prefix_matched: bool


@dataclass(frozen=True)
class ApproximateKVShareAnalysis:
    score: float
    label: str
    reason: str
    tenant_compatible: bool
    stage_compatible: bool
    canonical_prefix_matched: bool
    retrieval_overlap: float
    tool_overlap: float


@dataclass(frozen=True)
class KVEventStreamSummary:
    stored_hashes: tuple[Any, ...]
    removed_hashes: tuple[Any, ...]
    stored_event_count: int
    removed_event_count: int
    clear_event_count: int


@dataclass(frozen=True)
class KVRuntimeConsistencyReport:
    is_consistent: bool
    issues: tuple[str, ...]
    reference: KVEventStreamSummary
    candidate: KVEventStreamSummary
    missing_stores: tuple[Any, ...]
    extra_stores: tuple[Any, ...]
    missing_removes: tuple[Any, ...]
    extra_removes: tuple[Any, ...]


def evaluate_exact_kv_reuse(
    metadata: SharedExecutionMetadata | None,
    other: SharedExecutionMetadata | None,
) -> ExactKVReuseDecision:
    if metadata is None or other is None:
        return ExactKVReuseDecision(
            can_reuse=False,
            reason="missing_shared_execution_metadata",
            tenant_compatible=False,
            stage_compatible=False,
            canonical_prefix_matched=False,
        )

    canonical_prefix_matched = metadata.shares_canonical_prefix_with(other)
    if not canonical_prefix_matched:
        return ExactKVReuseDecision(
            can_reuse=False,
            reason="canonical_prefix_mismatch",
            tenant_compatible=False,
            stage_compatible=False,
            canonical_prefix_matched=False,
        )

    tenant_compatible = (
        metadata.tenant is None
        or other.tenant is None
        or metadata.tenant == other.tenant
    )
    if not tenant_compatible:
        return ExactKVReuseDecision(
            can_reuse=False,
            reason="tenant_mismatch",
            tenant_compatible=False,
            stage_compatible=False,
            canonical_prefix_matched=True,
        )

    stage_compatible = (
        metadata.stage_id is None
        or other.stage_id is None
        or metadata.stage_id == other.stage_id
    )
    if not stage_compatible:
        return ExactKVReuseDecision(
            can_reuse=False,
            reason="stage_mismatch",
            tenant_compatible=True,
            stage_compatible=False,
            canonical_prefix_matched=True,
        )

    return ExactKVReuseDecision(
        can_reuse=True,
        reason="canonical_prefix_match",
        tenant_compatible=True,
        stage_compatible=True,
        canonical_prefix_matched=True,
    )


def _normalized_overlap(
    left: Sequence[str],
    right: Sequence[str],
) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    if union == 0:
        return 0.0
    return intersection / union


def analyze_approximate_kv_sharing(
    metadata: SharedExecutionMetadata | None,
    other: SharedExecutionMetadata | None,
) -> ApproximateKVShareAnalysis:
    exact_decision = evaluate_exact_kv_reuse(metadata, other)
    if exact_decision.can_reuse:
        return ApproximateKVShareAnalysis(
            score=1.0,
            label="exact",
            reason="canonical_prefix_match",
            tenant_compatible=True,
            stage_compatible=True,
            canonical_prefix_matched=True,
            retrieval_overlap=1.0,
            tool_overlap=1.0,
        )

    if metadata is None or other is None:
        return ApproximateKVShareAnalysis(
            score=0.0,
            label="none",
            reason="missing_shared_execution_metadata",
            tenant_compatible=False,
            stage_compatible=False,
            canonical_prefix_matched=False,
            retrieval_overlap=0.0,
            tool_overlap=0.0,
        )

    tenant_compatible = (
        metadata.tenant is None
        or other.tenant is None
        or metadata.tenant == other.tenant
    )
    stage_compatible = (
        metadata.stage_id is None
        or other.stage_id is None
        or metadata.stage_id == other.stage_id
    )
    retrieval_overlap = _normalized_overlap(
        metadata.retrieval_signature,
        other.retrieval_signature,
    )
    tool_overlap = _normalized_overlap(
        metadata.tool_signature,
        other.tool_signature,
    )

    score = (
        0.45 * retrieval_overlap
        + 0.35 * tool_overlap
        + 0.10 * float(tenant_compatible)
        + 0.10 * float(stage_compatible)
    )
    if not tenant_compatible:
        score = min(score, 0.49)

    if score >= 0.75:
        label = "high"
    elif score >= 0.40:
        label = "partial"
    else:
        label = "none"

    if not tenant_compatible:
        reason = "tenant_mismatch"
    elif retrieval_overlap > 0 or tool_overlap > 0:
        reason = "signature_overlap"
    elif stage_compatible:
        reason = "stage_only_match"
    else:
        reason = "no_shared_signals"

    return ApproximateKVShareAnalysis(
        score=score,
        label=label,
        reason=reason,
        tenant_compatible=tenant_compatible,
        stage_compatible=stage_compatible,
        canonical_prefix_matched=False,
        retrieval_overlap=retrieval_overlap,
        tool_overlap=tool_overlap,
    )


def summarize_kv_cache_events(events: Sequence[object]) -> KVEventStreamSummary:
    stored_hashes: list[Any] = []
    removed_hashes: list[Any] = []
    stored_event_count = 0
    removed_event_count = 0
    clear_event_count = 0

    for event in events:
        if isinstance(event, BlockStored):
            stored_hashes.extend(event.block_hashes)
            stored_event_count += 1
        elif isinstance(event, BlockRemoved):
            removed_hashes.extend(event.block_hashes)
            removed_event_count += 1
        elif isinstance(event, AllBlocksCleared):
            clear_event_count += 1

    return KVEventStreamSummary(
        stored_hashes=tuple(stored_hashes),
        removed_hashes=tuple(removed_hashes),
        stored_event_count=stored_event_count,
        removed_event_count=removed_event_count,
        clear_event_count=clear_event_count,
    )


def _counter_diff(left: Counter[Any], right: Counter[Any]) -> tuple[Any, ...]:
    diff = left - right
    expanded: list[Any] = []
    for value, count in sorted(diff.items(), key=lambda item: repr(item[0])):
        expanded.extend([value] * count)
    return tuple(expanded)


def compare_kv_event_streams(
    reference_events: Sequence[object],
    candidate_events: Sequence[object],
) -> KVRuntimeConsistencyReport:
    reference = summarize_kv_cache_events(reference_events)
    candidate = summarize_kv_cache_events(candidate_events)

    reference_stores = Counter(reference.stored_hashes)
    candidate_stores = Counter(candidate.stored_hashes)
    reference_removes = Counter(reference.removed_hashes)
    candidate_removes = Counter(candidate.removed_hashes)

    missing_stores = _counter_diff(reference_stores, candidate_stores)
    extra_stores = _counter_diff(candidate_stores, reference_stores)
    missing_removes = _counter_diff(reference_removes, candidate_removes)
    extra_removes = _counter_diff(candidate_removes, reference_removes)

    issues: list[str] = []
    if reference.clear_event_count != candidate.clear_event_count:
        issues.append("clear_event_count_mismatch")
    if missing_stores or extra_stores:
        issues.append("stored_hash_mismatch")
    if missing_removes or extra_removes:
        issues.append("removed_hash_mismatch")

    return KVRuntimeConsistencyReport(
        is_consistent=not issues,
        issues=tuple(issues),
        reference=reference,
        candidate=candidate,
        missing_stores=missing_stores,
        extra_stores=extra_stores,
        missing_removes=missing_removes,
        extra_removes=extra_removes,
    )


def _normalize_signature(
    value: Sequence[str] | None,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if value is None:
        return ()

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"shared_execution.{field_name} must contain only strings, "
                f"got {type(item).__name__}"
            )
        text = item.strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _get_extra_payload(
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
) -> Any:
    if sampling_params is not None and sampling_params.extra_args is not None:
        return sampling_params.extra_args.get("shared_execution")
    if pooling_params is not None and pooling_params.extra_kwargs is not None:
        return pooling_params.extra_kwargs.get("shared_execution")
    return None


def extract_shared_execution_metadata(
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
) -> SharedExecutionMetadata | None:
    payload = _get_extra_payload(sampling_params, pooling_params)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(
            "shared_execution metadata must be a mapping passed via "
            "SamplingParams.extra_args['shared_execution'] or "
            "PoolingParams.extra_kwargs['shared_execution']"
        )

    tenant = payload.get("tenant")
    if tenant is not None and not isinstance(tenant, str):
        raise ValueError("shared_execution.tenant must be a string")

    canonical_prefix_key = payload.get("canonical_prefix_key")
    if canonical_prefix_key is None:
        canonical_prefix_key = payload.get("prefix_key")
    if canonical_prefix_key is not None and not isinstance(canonical_prefix_key, str):
        raise ValueError("shared_execution.canonical_prefix_key must be a string")

    stage_id = payload.get("stage_id")
    if stage_id is not None and not isinstance(stage_id, str):
        raise ValueError("shared_execution.stage_id must be a string")

    priority_hint = payload.get("priority_hint")
    if priority_hint is not None and not isinstance(priority_hint, int):
        raise ValueError("shared_execution.priority_hint must be an int")

    retrieval_signature = _normalize_signature(
        payload.get("retrieval_signature") or payload.get("retrieval_keys"),
        field_name="retrieval_signature",
    )
    tool_signature = _normalize_signature(
        payload.get("tool_signature") or payload.get("tool_calls"),
        field_name="tool_signature",
    )

    metadata = SharedExecutionMetadata(
        tenant=tenant,
        canonical_prefix_key=canonical_prefix_key.strip()
        if isinstance(canonical_prefix_key, str) and canonical_prefix_key.strip()
        else None,
        retrieval_signature=retrieval_signature,
        tool_signature=tool_signature,
        stage_id=stage_id.strip() if isinstance(stage_id, str) and stage_id.strip() else None,
        priority_hint=priority_hint,
    )

    if (
        metadata.tenant is None
        and metadata.canonical_prefix_key is None
        and not metadata.retrieval_signature
        and not metadata.tool_signature
        and metadata.stage_id is None
        and metadata.priority_hint is None
    ):
        return None
    return metadata