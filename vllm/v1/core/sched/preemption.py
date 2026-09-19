# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extension API for selecting requests to preempt under KV pressure."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.plugins.evidence import EvidenceScope

logger = init_logger(__name__)

PreemptionSchedulingPolicy = Literal["fcfs", "priority"]
PREEMPTION_POLICY_API_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class PreemptionCandidate:
    """Immutable request data available to a preemption policy."""

    request_id: str
    priority: int
    arrival_time: float
    num_prompt_tokens: int
    num_output_tokens: int
    num_computed_tokens: int
    num_preemptions: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class PreemptionContext:
    """Immutable snapshot of one scheduler preemption decision.

    ``requesting_request_id`` identifies the running request whose KV
    allocation just failed. Selecting it resets its computed KV state, so a
    policy that cannot establish a material benefit should return ``None``.
    ``builtin_victim_id`` lets policies compare against vLLM's stable fallback
    without copying FCFS/priority ordering rules. It is optional only for
    source compatibility with callers that constructed API 1.0 contexts.
    """

    candidates: tuple[PreemptionCandidate, ...]
    scheduling_policy: PreemptionSchedulingPolicy
    requesting_request_id: str
    kv_cache_usage: float
    now: float
    builtin_victim_id: str | None = None


@runtime_checkable
class PreemptionPolicy(Protocol):
    """Policy contract for request preemption.

    Returning ``None`` delegates the decision to vLLM's built-in policy.
    Implementations must return an ID from ``context.candidates`` and must not
    retain references to scheduler-owned mutable state.
    """

    def select_victim(self, context: PreemptionContext) -> str | None:
        """Return the request ID to preempt, or ``None`` to abstain."""
        ...


@dataclass(slots=True)
class PreemptionPolicyStats:
    """Cumulative policy invocation and failover counters."""

    policy_name: str = "builtin"
    enabled: bool = False
    calls: int = 0
    selections: int = 0
    abstentions: int = 0
    failures: int = 0
    invalid_selections: int = 0


def _builtin_victim_id(context: PreemptionContext) -> str:
    if not context.candidates:
        raise ValueError("cannot select a preemption victim from an empty set")
    candidate_ids = {candidate.request_id for candidate in context.candidates}
    if context.builtin_victim_id in candidate_ids:
        return context.builtin_victim_id
    if context.scheduling_policy == "priority":
        return max(
            context.candidates,
            key=lambda candidate: (candidate.priority, candidate.arrival_time),
        ).request_id
    return context.candidates[-1].request_id


def _policy_name(policy: PreemptionPolicy) -> str:
    policy_type = type(policy)
    return f"{policy_type.__module__}.{policy_type.__qualname__}"


def _emit_policy_evidence(
    policy_name: str,
    event: str,
    outcome: str,
    scope: EvidenceScope,
    occurrence_id: int | str,
    controller_instance_id: str,
) -> None:
    try:
        from vllm.plugins.evidence import _emit_host_event

        _emit_host_event(
            event,
            "vllm.preemption_policy",
            policy_name,
            policy_name,
            detail=f"engine-core.scheduler:{outcome}",
            scope=scope,
            occurrence_id=occurrence_id,
            controller_instance_id=controller_instance_id,
            observation_kind=(
                "scheduler_dispatch" if event == "invoked" else "scheduler_resolution"
            ),
        )
    except Exception:
        # Evidence is observational and records/logs its own first sink failure.
        # Never let a strict or broken sink perturb the scheduler hot path.
        return


def _load_policy(vllm_config: VllmConfig) -> PreemptionPolicy | None:
    configured: Any = vllm_config.scheduler_config.preemption_policy
    if configured is None:
        return None
    implementation = (
        resolve_obj_by_qualname(configured)
        if isinstance(configured, str)
        else configured
    )
    factory = getattr(implementation, "from_vllm_config", None)
    if callable(factory):
        policy = factory(vllm_config)
    elif isinstance(implementation, type):
        policy = implementation()
    else:
        policy = implementation
    if not isinstance(policy, PreemptionPolicy):
        raise TypeError(
            "preemption_policy must resolve to an object implementing PreemptionPolicy"
        )
    return policy


class PreemptionPolicyController:
    """Validate an external policy and fail over permanently on a fault.

    The scheduler owns each controller from one thread. Evidence has its own
    process-wide locked allocators, but policy selection is deliberately not a
    concurrent API.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        policy = _load_policy(vllm_config)
        self._policy = policy
        self._evidence_scope: EvidenceScope | None = None
        self._controller_instance_id: str | None = None
        self._owner_thread_id = threading.get_ident()
        self.stats = PreemptionPolicyStats(
            policy_name=_policy_name(policy) if policy is not None else "builtin",
            enabled=policy is not None,
        )
        logger.info(
            "Preemption policy initialized: policy=%s enabled=%s",
            self.stats.policy_name,
            self.stats.enabled,
        )
        if policy is not None:
            self._capture_evidence_scope()

    def select_victim(self, context: PreemptionContext) -> str:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError(
                "PreemptionPolicyController must be used by its scheduler thread"
            )
        policy = self._policy
        if policy is None:
            return _builtin_victim_id(context)

        self.stats.calls += 1
        try:
            selected_id = policy.select_victim(context)
        except Exception:
            self._emit_invocation("exception")
            self._disable_after_failure("raised an exception", exc_info=True)
            return _builtin_victim_id(context)

        if selected_id is None:
            self.stats.abstentions += 1
            self._emit_invocation("abstained")
            return _builtin_victim_id(context)

        candidate_ids = {candidate.request_id for candidate in context.candidates}
        if selected_id not in candidate_ids:
            self.stats.invalid_selections += 1
            self._emit_invocation("invalid")
            self._disable_after_failure(f"returned unknown request ID {selected_id!r}")
            return _builtin_victim_id(context)

        self.stats.selections += 1
        self._emit_invocation("selected")
        return selected_id

    def _emit_invocation(self, outcome: str) -> None:
        self._refresh_evidence_scope()
        if self._evidence_scope is None or self._controller_instance_id is None:
            return
        try:
            from vllm.plugins.evidence import allocate_invocation_sequence

            invocation_seq = allocate_invocation_sequence(self._evidence_scope)
        except Exception:
            return
        _emit_policy_evidence(
            self.stats.policy_name,
            "invoked",
            outcome,
            self._evidence_scope,
            invocation_seq,
            self._controller_instance_id,
        )

    def _refresh_evidence_scope(self) -> None:
        if self._evidence_scope is None:
            return
        try:
            from vllm.plugins.evidence import scope_is_current

            if scope_is_current(self._evidence_scope):
                return
        except Exception:
            pass
        self._evidence_scope = None
        self._controller_instance_id = None
        self._capture_evidence_scope()

    def _capture_evidence_scope(self) -> None:
        try:
            from vllm.plugins.evidence import (
                allocate_controller_instance,
                capture_scope,
                observer_configured,
            )

            if not observer_configured():
                return
            scope = capture_scope(expected_role="engine-core-scheduler")
            controller_instance_id = allocate_controller_instance(scope)
        except Exception as exc:
            logger.warning_once(
                "Preemption policy evidence scope is unavailable: policy=%s error=%r",
                self.stats.policy_name,
                exc,
            )
            return
        self._evidence_scope = scope
        self._controller_instance_id = controller_instance_id
        _emit_policy_evidence(
            self.stats.policy_name,
            "resolved",
            "protocol-validated",
            scope,
            f"controller:{controller_instance_id}",
            controller_instance_id,
        )

    def export_stats(self) -> dict[str, str | int | bool]:
        """Return a serialization-safe cumulative stats snapshot."""
        return asdict(self.stats)

    def _disable_after_failure(self, reason: str, *, exc_info: bool = False) -> None:
        self.stats.failures += 1
        self.stats.enabled = False
        self._policy = None
        logger.error(
            "Preemption policy %s failed (%s); disabling it and restoring "
            "the built-in policy for this engine process",
            self.stats.policy_name,
            reason,
            exc_info=exc_info,
        )


__all__ = [
    "PREEMPTION_POLICY_API_VERSION",
    "PreemptionCandidate",
    "PreemptionContext",
    "PreemptionPolicy",
    "PreemptionPolicyController",
    "PreemptionPolicyStats",
]
