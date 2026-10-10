"""Prompt-free projections at the scheduler's pre-decision boundary.

The projection in this module is deliberately a pure function of scheduler
state that already exists *before* the scheduling decision: queue membership,
received prompt/output counters, the received generation cap, request
priority/arrival time, declared request objectives and the KV block pool.  It
never reads a formed batch, a response, or a terminal outcome.

``CausalPreDecisionObserver`` is the bounded, environment-gated carrier for
those projections.  It follows the same contract as the other scheduler
observers: disabled by default, request-prefix scoped, cursor-readable, and
capacity bounded.
"""
from __future__ import annotations

import os
import time
from collections import deque
from copy import deepcopy
from typing import Any

SCHEMA = "anise.scheduler-pre-decision-state.v1"
ENABLED_ENV = "QBI_CAUSAL_PRE_DECISION_OBSERVER"
CAPACITY_ENV = "QBI_CAUSAL_PRE_DECISION_OBSERVER_CAPACITY"
PREFIX_ENV = "QBI_CAUSAL_PRE_DECISION_REQUEST_PREFIX"
DEFAULT_REQUEST_PREFIX = "chatcmpl-anise-causal-"
DEFAULT_CAPACITY = 8192
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _resident_prefix_tokens(request, kv_cache_manager, pending_prefill):
    """Exact cache-aware prefix length for this request, read-only.

    It asks the coordinator the same question the scheduler asks later in the step
    (``find_longest_cache_hit`` over the request's own ``block_hashes``, which were
    computed when the request was constructed), so the number is exactly what the
    upcoming lookup will reuse.  The find phase allocates nothing and touches
    nothing -- allocation happens later, in ``allocate_slots`` -- which is what
    makes it safe to ask before the decision.

    Two earlier designs are recorded as dead ends in
    ``work/replay_validity/FIX_CAUSAL_PROJECTION_CACHE_AWARENESS_ATTEMPT_20260922_V25_V26.json``:
    testing ``block_hash in pool.cached_block_hash_to_block`` (that index is a
    ``BlockHashToBlockMap``, and the resulting ``TypeError`` killed EngineCore),
    and asking the pool for every kv-cache group at once (this carrier is a hybrid
    model, so the group-wise requirement never matches and the count was always 0).

    Returns ``None`` when the carrier does not expose the query (so the projection
    never invents a number), and also on any failure: this runs inside the
    scheduler's step loop for an *optional* diagnostic observer, so it must never
    be able to raise.
    """

    if kv_cache_manager is None:
        return None
    if getattr(kv_cache_manager, "enable_caching", None) is False:
        # Prefix caching is off for the whole carrier: nothing can be resident.
        return 0
    if getattr(request, "skip_reading_prefix_cache", False):
        # The request carries a policy that bypasses the cache, so the upcoming
        # lookup will not reuse anything even if blocks are resident.
        return 0
    coordinator = getattr(kv_cache_manager, "coordinator", None)
    if coordinator is None:
        return None
    if pending_prefill <= 0:
        # A request with nothing left to prefill cannot reuse a prefix, so the
        # query is skipped entirely: this keeps the projection O(1) for the decode
        # requests that make up most of every step.
        return 0
    hashes = getattr(request, "block_hashes", None)
    if not hashes:
        return 0
    try:
        max_cache_hit_length = max(0, int(getattr(request, "num_tokens", 0)) - 1)
        _blocks, hit_tokens = coordinator.find_longest_cache_hit(
            list(hashes), max_cache_hit_length
        )
    except Exception:  # noqa: BLE001 -- never let the observer raise into the step loop
        return None
    if isinstance(hit_tokens, bool) or not isinstance(hit_tokens, int):
        return None
    return min(hit_tokens, pending_prefill)


def project_causal_state(scheduler, *, observed_at_ns, wall_time):
    """Project the pre-decision scheduler state for one scheduling step."""
    waiting = list(scheduler.waiting) + list(scheduler.skipped_waiting)
    running = list(scheduler.running)
    kv_manager = getattr(scheduler, "kv_cache_manager", None)
    members = []
    for request in running + waiting:
        in_wait = request in waiting
        output = request.num_output_tokens
        prefill = max(0, request.num_prompt_tokens - request.num_computed_tokens)
        try:
            cached_prefix = _resident_prefix_tokens(request, kv_manager, prefill)
        except Exception:  # noqa: BLE001 -- belt and braces around the observer
            cached_prefix = None
        # This is remaining CAPACITY under the received generation cap, not a
        # prediction of EOS or actual future service.
        remaining = max(0, request.max_tokens - output)
        extra = getattr(request.sampling_params, "extra_args", None) or {}
        requested_decode = request.max_tokens
        members.append(
            {
                "request_id": request.request_id,
                "priority": request.priority,
                "prompt_tokens": request.num_prompt_tokens,
                "uncomputed_prefill_tokens": prefill,
                # Cache-aware view of the same quantity: what would remain to
                # prefill if this request's resident prefix were reused.  Both
                # numbers are reported so a consumer can tell "no cache" from
                # "the cache was not consulted yet".
                "cached_prefix_tokens": cached_prefix,
                "cache_aware_uncomputed_prefill_tokens": (
                    None if cached_prefix is None else max(0, prefill - cached_prefix)
                ),
                "remaining_decode_cap_tokens": remaining,
                "requested_decode_tokens": requested_decode,
                "remaining_service_cap_tokens": prefill + remaining,
                "output_tokens_so_far": output,
                "waiting": in_wait,
                "arrival_time": request.arrival_time,
                "wait_since_arrival_ms": (
                    max(0.0, wall_time - request.arrival_time) * 1000
                    if in_wait
                    else None
                ),
                "active_decode": request in running and output > 0 and prefill == 0,
                "slo_deadline_class": extra.get(
                    "qbi_slo_deadline_class", "undeclared"
                ),
                "fairness_requirement": extra.get(
                    "qbi_fairness_requirement", "undeclared"
                ),
                "ttft_objective_ms": extra.get("qbi_ttft_objective_ms"),
                "itl_objective_ms": extra.get("qbi_itl_objective_ms"),
            }
        )
    kv = getattr(scheduler, "kv_cache_manager", None)
    pressure = getattr(kv, "usage", None)
    return {
        "schema": SCHEMA,
        "observed_at_ns": observed_at_ns,
        "clock_id": "monotonic_ns",
        "config_epoch": scheduler._runtime_config_epoch,
        "queue_policy": scheduler.policy.value,
        "members": members,
        "queue_depth": len(waiting),
        "active_request_count": len(running),
        "active_decode_count": sum(m["active_decode"] for m in members),
        "prefill_tokens": sum(m["uncomputed_prefill_tokens"] for m in members),
        # Cache-aware totals for the whole step.  ``cached_prefix_tokens`` is the
        # sum of the resident-prefix counts above; ``cache_aware_prefill_tokens``
        # is what the same step would have to prefill if those prefixes were
        # reused.  Both are None-free only when the carrier exposes its cached
        # block index; otherwise they are reported as null rather than invented.
        "cached_prefix_tokens": (
            None
            if any(m["cached_prefix_tokens"] is None for m in members)
            else sum(m["cached_prefix_tokens"] for m in members)
        ),
        "cache_aware_prefill_tokens": (
            None
            if any(m["cache_aware_uncomputed_prefill_tokens"] is None for m in members)
            else sum(m["cache_aware_uncomputed_prefill_tokens"] for m in members)
        ),
        "chunk_budget_tokens": scheduler.max_num_scheduled_tokens,
        "kv_pressure_ratio": pressure,
        "fairness_debt_ms": sum(
            m["wait_since_arrival_ms"] for m in members if m["waiting"]
        ),
        "preemption_context_switch_cost_ms": (
            getattr(scheduler, "_qbi_preemption_elapsed_ns", 0) / 1e6
        ),
        "preemption_count": getattr(scheduler, "_qbi_preemption_count", 0),
        "previous_schedule_overhead_ms": getattr(
            scheduler, "_qbi_previous_schedule_ms", None
        ),
        # Measured pure-prefill device cost per scheduled token, carried with the
        # instant it was measured so the receipt can be checked against the
        # decision clock.  Absent until a pure-prefill step has been observed.
        "prefill_ms_per_token": (
            getattr(scheduler, "_qbi_prefill_cost_sample", None) or {}
        ).get("ms_per_scheduled_token"),
        "prefill_ms_per_token_observed_at_ns": (
            getattr(scheduler, "_qbi_prefill_cost_sample", None) or {}
        ).get("observed_at_ns"),
        "prefill_ms_per_token_source": (
            getattr(scheduler, "_qbi_prefill_cost_sample", None) or {}
        ).get("source"),
        "semantics": {
            "remaining_service": (
                "received remaining prefill + remaining max_tokens cap; "
                "not predicted EOS"
            ),
            "fairness_debt": (
                "sum of age of currently waiting requests; running service excluded"
            ),
            "preemption_cost": (
                "measured cumulative CPU time in actual _preempt_request calls "
                "since boot; zero when none executed"
            ),
            "previous_schedule_overhead_ms": (
                "last completed schedule CPU duration in ms, before current decision"
            ),
            "prefill_ms_per_token": (
                "measured engine-step device time per scheduled token from the "
                "most recent step whose scheduled tokens were all prefill tokens; "
                "carries its own observed_at_ns because it precedes this decision; "
                "absent (not zero) when no pure-prefill step has been observed"
            ),
            "objectives": (
                "declared request contract; never runtime latency observations"
            ),
        },
    }


class CausalPreDecisionObserver:
    """Retain bounded pre-decision projections for owned request IDs only."""

    def __init__(
        self, *, enabled: bool, capacity: int, request_prefix: str
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("causal pre-decision enabled must be boolean")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("causal pre-decision capacity must be an integer")
        if not 128 <= capacity <= 65536:
            raise ValueError(
                "causal pre-decision capacity must be in [128, 65536]"
            )
        normalized_prefix = str(request_prefix).strip()
        if not normalized_prefix:
            raise ValueError("causal pre-decision request prefix must not be empty")
        self.enabled = enabled
        self.capacity = capacity
        self.request_prefix = normalized_prefix
        self.sequence = 0
        self.dropped_receipts = 0
        self._receipts: deque[dict[str, Any]] = deque(maxlen=capacity)

    @classmethod
    def from_environment(cls) -> "CausalPreDecisionObserver":
        raw_enabled = os.environ.get(ENABLED_ENV, "0").strip().lower()
        if raw_enabled not in _TRUE_VALUES | _FALSE_VALUES:
            raise ValueError(f"{ENABLED_ENV} must be a boolean flag")
        capacity_text = os.environ.get(CAPACITY_ENV, str(DEFAULT_CAPACITY)).strip()
        if not capacity_text.isdecimal():
            raise ValueError(f"{CAPACITY_ENV} must be an integer")
        return cls(
            enabled=raw_enabled in _TRUE_VALUES,
            capacity=int(capacity_text),
            request_prefix=os.environ.get(PREFIX_ENV, DEFAULT_REQUEST_PREFIX),
        )

    def observe(self, scheduler, *, observed_at_ns, wall_time) -> None:
        """Record one pre-decision projection when an owned request is present.

        The projection itself is only computed when at least one request in the
        current scheduler view matches the observer prefix, so an unowned
        workload pays a single string scan and no allocation.
        """
        if not self.enabled:
            return
        candidates = (
            *scheduler.running,
            *scheduler.waiting,
            *scheduler.skipped_waiting,
        )
        if not any(
            str(getattr(request, "request_id", "")).startswith(self.request_prefix)
            for request in candidates
        ):
            return
        receipt = project_causal_state(
            scheduler, observed_at_ns=observed_at_ns, wall_time=wall_time
        )
        self.sequence += 1
        if len(self._receipts) == self.capacity:
            self.dropped_receipts += 1
        receipt["sequence"] = self.sequence
        # ``observed_at_ns`` is taken by the caller immediately before this
        # projection and the scheduling decision happens strictly after it.
        receipt["decision_at_ns"] = time.monotonic_ns()
        self._receipts.append(receipt)

    def state(self, after_sequence: int = 0) -> dict[str, Any]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("causal pre-decision cursor must be an integer")
        if after_sequence < 0:
            raise ValueError("causal pre-decision cursor must be non-negative")
        receipts = [
            deepcopy(receipt)
            for receipt in self._receipts
            if int(receipt["sequence"]) > after_sequence
        ]
        oldest = int(self._receipts[0]["sequence"]) if self._receipts else 0
        return {
            "schema": "qbi.causal-pre-decision-scheduler-state.v1",
            "observer_enabled": self.enabled,
            "request_prefix": self.request_prefix,
            "capacity": self.capacity,
            "oldest_available_sequence": oldest,
            "latest_sequence": self.sequence,
            "dropped_receipts": self.dropped_receipts,
            "receipts": receipts,
        }


__all__ = [
    "SCHEMA",
    "CausalPreDecisionObserver",
    "project_causal_state",
]
