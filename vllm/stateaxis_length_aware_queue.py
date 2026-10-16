# SPDX-License-Identifier: Apache-2.0
"""Default-off StateAxis cold-prefill length-aware queue contract."""

from dataclasses import dataclass
from typing import Any


MOD_ID = "org.vllm-hust.stateaxis-length-aware-queue"


@dataclass(frozen=True, slots=True)
class LengthAwareQueueExperiment:
    short_max_tokens: int
    medium_max_tokens: int
    aging_ms: int
    max_queued_requests: int
    max_queued_tokens: int


@dataclass(frozen=True, slots=True)
class QueueItem:
    request_id: str
    prompt_tokens: int
    arrival_time: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class QueuePlan:
    request_ids: tuple[str, ...]
    reordered: bool
    aged_promotions: int
    fallback_reason: str | None = None


def parse_length_aware_queue_experiment(
    additional_config: dict[str, Any] | object,
    *,
    device_type: str,
) -> LengthAwareQueueExperiment | None:
    """Validate the exact ECPA-owned queue contract, or return OFF."""

    if not isinstance(additional_config, dict):
        return None
    raw = additional_config.get("stateaxis_length_aware_queue")
    if raw is None:
        return None
    binding = additional_config.get("stateaxis_mod")
    if not isinstance(binding, dict) or binding.get("mod_id") != MOD_ID:
        raise ValueError(
            "stateaxis_length_aware_queue requires its matching StateAxis MOD binding"
        )
    if additional_config.get("experiment_mode") is not True:
        raise ValueError("stateaxis_length_aware_queue requires experiment_mode")
    manifest_sha256 = binding.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise ValueError(
            "stateaxis_length_aware_queue requires a SHA-256 manifest binding"
        )
    expected = {
        "enabled": True,
        "telemetry_enabled": True,
        "authority_enabled": True,
        "scope": "cold-prefill-no-state",
        "short_max_tokens": 128,
        "medium_max_tokens": 512,
        "aging_ms": 250,
        "max_queued_requests": 64,
        "max_queued_tokens": 1_048_576,
        "atomic_full_plan_reorder": True,
        "trace_identity_aligned": True,
        "stateful_or_decode_fallback": True,
        "monotonic_clock_fail_closed": True,
        "fail_closed": True,
    }
    if raw != expected:
        raise ValueError("stateaxis_length_aware_queue requires the reviewed contract")
    if device_type != "npu":
        raise ValueError("stateaxis_length_aware_queue currently supports Ascend NPU")
    return LengthAwareQueueExperiment(
        short_max_tokens=128,
        medium_max_tokens=512,
        aging_ms=250,
        max_queued_requests=64,
        max_queued_tokens=1_048_576,
    )


def plan_length_aware_order(
    experiment: LengthAwareQueueExperiment,
    items: list[QueueItem],
    *,
    now: float,
) -> QueuePlan:
    """Plan one bounded atomic reorder without mutating the host queue."""

    original = tuple(item.request_id for item in items)
    if len(items) > experiment.max_queued_requests:
        return QueuePlan(original, False, 0, "request_bound")
    if sum(item.prompt_tokens for item in items) > experiment.max_queued_tokens:
        return QueuePlan(original, False, 0, "token_bound")
    if any(not item.eligible for item in items):
        return QueuePlan(original, False, 0, "ineligible_request")
    if any(now < item.arrival_time for item in items):
        return QueuePlan(original, False, 0, "clock_regression")

    def length_class(prompt_tokens: int) -> int:
        if prompt_tokens <= experiment.short_max_tokens:
            return 0
        if prompt_tokens <= experiment.medium_max_tokens:
            return 1
        return 2

    aged = {
        item.request_id
        for item in items
        if (now - item.arrival_time) * 1000 >= experiment.aging_ms
    }
    ordered = sorted(
        items,
        key=lambda item: (
            0 if item.request_id in aged else 1,
            item.arrival_time
            if item.request_id in aged
            else length_class(item.prompt_tokens),
            item.arrival_time,
        ),
    )
    request_ids = tuple(item.request_id for item in ordered)
    aged_promotions = sum(
        index < original.index(request_id)
        for index, request_id in enumerate(request_ids)
        if request_id in aged
    )
    return QueuePlan(
        request_ids,
        request_ids != original,
        aged_promotions,
    )
