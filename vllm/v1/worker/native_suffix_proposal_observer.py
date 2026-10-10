"""Bounded, answer-blind Suffix proposal receipts captured before verification."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from typing import Any


class NativeSuffixProposalObserver:
    """Capture the exact suffix-cache candidate state at the S4 decision point."""

    def __init__(
        self,
        *,
        enabled: bool,
        capacity: int,
        request_prefix: str,
        worker_rank: int | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("native Suffix observer enabled must be boolean")
        if type(capacity) is not int or not 128 <= capacity <= 65536:
            raise ValueError("native Suffix observer capacity must be in [128, 65536]")
        if not str(request_prefix).strip():
            raise ValueError("native Suffix request prefix must not be empty")
        if worker_rank is not None and (type(worker_rank) is not int or worker_rank < 0):
            raise ValueError("native Suffix worker rank must be non-negative")
        self.enabled = enabled
        self.capacity = capacity
        self.request_prefix = str(request_prefix).strip()
        self.worker_rank = worker_rank
        self.sequence = 0
        self.dropped_receipts = 0
        self._receipts: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @classmethod
    def from_environment(
        cls, *, worker_rank: int | None = None
    ) -> NativeSuffixProposalObserver:
        raw = os.environ.get("QBI_NATIVE_SUFFIX_PROPOSAL_OBSERVER", "0").strip().lower()
        if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("QBI_NATIVE_SUFFIX_PROPOSAL_OBSERVER must be a boolean flag")
        capacity_text = os.environ.get(
            "QBI_NATIVE_SUFFIX_PROPOSAL_OBSERVER_CAPACITY", "8192"
        ).strip()
        if not capacity_text.isdecimal():
            raise ValueError("QBI_NATIVE_SUFFIX_PROPOSAL_OBSERVER_CAPACITY must be an integer")
        return cls(
            enabled=raw in {"1", "true", "yes", "on"},
            capacity=int(capacity_text),
            request_prefix=os.environ.get(
                "QBI_NATIVE_SUFFIX_PROPOSAL_REQUEST_PREFIX",
                "chatcmpl-anise-suffix-",
            ),
            worker_rank=worker_rank,
        )

    def record_pre_verification_decision(
        self,
        *,
        request_id: str,
        request_slot: int,
        context_token_count: int,
        prompt_token_count: int,
        sampled_token_count: int,
        pattern_token_count: int,
        match_length: int,
        draft_token_count: int,
        draft_score: float,
        draft_probabilities: Sequence[float],
        proposer_elapsed_ns: int,
        active_request_count: int,
        config_epoch: int | None,
        observed_at_ns: int | None = None,
    ) -> int | None:
        """Freeze proposal features before the scheduler admits verification."""
        request_id = str(request_id)
        if not self.enabled or not request_id.startswith(self.request_prefix):
            return None
        integer_values = {
            "request_slot": request_slot,
            "context_token_count": context_token_count,
            "prompt_token_count": prompt_token_count,
            "sampled_token_count": sampled_token_count,
            "pattern_token_count": pattern_token_count,
            "match_length": match_length,
            "draft_token_count": draft_token_count,
            "proposer_elapsed_ns": proposer_elapsed_ns,
            "active_request_count": active_request_count,
        }
        if any(type(value) is not int or value < 0 for value in integer_values.values()):
            raise ValueError("native Suffix proposal receipt contains an invalid count")
        if match_length > pattern_token_count:
            raise ValueError("native Suffix match length exceeds its input pattern")
        if type(draft_score) not in (int, float) or not math.isfinite(float(draft_score)):
            raise ValueError("native Suffix draft score must be finite")
        probabilities = [float(value) for value in draft_probabilities]
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("native Suffix draft probabilities must be in [0, 1]")
        if len(probabilities) != draft_token_count:
            raise ValueError("native Suffix probability count differs from draft token count")
        stamp = time.monotonic_ns() if observed_at_ns is None else observed_at_ns
        if type(stamp) is not int or stamp < 0:
            raise ValueError("native Suffix observation timestamp is invalid")
        epoch = config_epoch if type(config_epoch) is int and config_epoch >= 0 else None
        probability_min = min(probabilities) if probabilities else None
        probability_mean = sum(probabilities) / len(probabilities) if probabilities else None
        with self._lock:
            self.sequence += 1
            if len(self._receipts) == self.capacity:
                self.dropped_receipts += 1
            self._receipts.append(
                {
                    "schema": "qbi.native-suffix-pre-verification.v1",
                    "sequence": self.sequence,
                    "decision_id": f"{request_id}:suffix:{self.sequence}",
                    "worker_rank": self.worker_rank,
                    "provider_request_id": request_id,
                    "request_slot": request_slot,
                    "observation_point": "s4_suffix_cache_probe_before_verification_admission",
                    "observation_is_execution_completion": False,
                    "clock_id": "host.monotonic_ns",
                    "observed_at_ns": stamp,
                    "decision_at_ns": stamp,
                    "config_epoch": epoch,
                    "epoch_status": "observed" if epoch is not None else "missing",
                    "features": {
                        "suffix_cache_match_length": match_length,
                        "expected_draft_tokens": draft_token_count,
                        "verification_work_units": draft_token_count,
                        "candidate_confidence_sum": float(draft_score),
                        "candidate_probability_min": probability_min,
                        "candidate_probability_mean": probability_mean,
                        "proposer_elapsed_ns": proposer_elapsed_ns,
                        "pattern_token_count": pattern_token_count,
                        "context_token_count": context_token_count,
                        "prompt_token_count": prompt_token_count,
                        "sampled_token_count": sampled_token_count,
                        "generated_tokens_before_decision": (
                            max(0, context_token_count - prompt_token_count)
                            + sampled_token_count
                        ),
                        "active_request_count": active_request_count,
                    },
                }
            )
            return self.sequence

    def state(self, after_sequence: int = 0) -> dict[str, Any]:
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("native Suffix sequence cursor must be non-negative")
        with self._lock:
            return {
                "schema": "qbi.native-suffix-worker-state.v1",
                "status": "READY" if self.enabled else "NOT_READY",
                "observer_enabled": self.enabled,
                "worker_rank": self.worker_rank,
                "request_prefix": self.request_prefix,
                "capacity": self.capacity,
                "oldest_available_sequence": (
                    int(self._receipts[0]["sequence"]) if self._receipts else 0
                ),
                "latest_sequence": self.sequence,
                "dropped_receipts": self.dropped_receipts,
                "scope": "s4_pre_verification",
                "receipts": [
                    deepcopy(row)
                    for row in self._receipts
                    if int(row["sequence"]) > after_sequence
                ],
                "field_status": {
                    "suffix_cache_match_length": "READY",
                    "expected_draft_tokens": "observed_pre_verification",
                    "verification_work_units": "observed_pre_verification",
                    "candidate_score": "READY",
                    "candidate_probabilities": "READY",
                    "proposer_elapsed_ns": "READY",
                    "config_epoch": "READY" if self.enabled else "NOT_READY",
                },
                "forbidden_substitutes": [
                    "accepted_token_count",
                    "completion_latency",
                    "task_score",
                    "terminal_outcome",
                ],
            }


__all__ = ["NativeSuffixProposalObserver"]
