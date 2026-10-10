# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded receipts for the native request-local repetition detector."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import deque
from typing import Any


class RepetitionDetectionObserver:
    """Record detector checks and terminal effects without changing its algorithm."""

    def __init__(self, *, capacity: int = 8192) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("repetition observer capacity must be an integer")
        if not 128 <= capacity <= 65536:
            raise ValueError("repetition observer capacity must be in [128, 65536]")
        self.capacity = capacity
        self.sequence = 0
        self.dropped_receipts = 0
        self._checks: dict[str, int] = {}
        self._configs: dict[str, dict[str, int]] = {}
        self._receipts: deque[dict[str, Any]] = deque(maxlen=capacity)
        # The scheduler records this request-local snapshot immediately before
        # calling check_stop.  Terminal receipts may only bind this snapshot;
        # they must never reconstruct it from a response or terminal state.
        self._pending_pre_stop: dict[str, dict[str, Any]] = {}
        self._pre_stop_events: deque[dict[str, Any]] = deque(maxlen=capacity)

    @staticmethod
    def _repetition_features(
        token_ids: Any, params: Any
    ) -> tuple[int, int, float]:
        """Measure the current generated suffix using the native token IDs.

        The values describe the suffix at the decision boundary.  They are
        deliberately computed before ``check_stop`` mutates request status;
        no response text or terminal receipt participates in this function.
        """
        if not isinstance(token_ids, (list, tuple)):
            raise ValueError("pre-stop observer requires request-local token IDs")
        tokens = list(token_ids)
        max_size = int(params.max_pattern_size)
        min_size = max(1, int(params.min_pattern_size))
        min_count = int(params.min_count)
        if max_size <= 0 or min_size > max_size or min_count < 2:
            raise ValueError("pre-stop observer received an invalid detector")
        best_length = 0
        best_count = 1
        # Keep the same ascending pattern preference as check_sequence_repetition
        # when a detector trigger is present.  For a non-triggering suffix we
        # still expose the strongest currently observed repeated tail.
        for pattern_length in range(min_size, max_size + 1):
            if len(tokens) < pattern_length * 2:
                continue
            tail = tokens[-pattern_length:]
            count = 1
            while len(tokens) >= (count + 1) * pattern_length:
                start = -(count + 1) * pattern_length
                if tokens[start : start + pattern_length] != tail:
                    break
                count += 1
            if count >= min_count:
                best_length, best_count = pattern_length, count
                break
            if count > best_count:
                best_length, best_count = pattern_length, count
        if best_count < 2:
            best_length = 0
        generated = len(tokens)
        growth = (best_count - 1) / max(1, generated)
        return best_length, best_count, float(growth)

    def record_pre_stop(
        self,
        *,
        request_id: str,
        params: Any,
        generated_token_ids: Any,
        generated_token_count: int,
        stop_contract: str,
        clock_id: str = "engine-monotonic-ns-v1",
    ) -> dict[str, Any]:
        """Freeze a provider-request-bound event immediately before stop.

        ``record_finish`` receives and persists this object.  It does not
        recalculate any feature from the response after the stop decision.
        """
        request_id = str(request_id)
        config = self._config(params)
        if self._configs.get(request_id) != config:
            raise RuntimeError("pre-stop event lacks a matching checked config")
        if type(generated_token_count) is not int or generated_token_count < 1:
            raise ValueError("pre-stop generated token count must be positive")
        if not isinstance(generated_token_ids, (list, tuple)) or generated_token_count != len(generated_token_ids):
            raise ValueError("pre-stop token count is not request-local output length")
        if not isinstance(stop_contract, str) or not stop_contract.strip():
            raise ValueError("pre-stop stop contract is missing")
        if not isinstance(clock_id, str) or not clock_id.strip():
            raise ValueError("pre-stop clock identity is missing")
        current_length, count, growth = self._repetition_features(
            generated_token_ids, params
        )
        observed_at_ns = time.monotonic_ns()
        tail_count = min(
            generated_token_count,
            int(params.max_pattern_size) * max(2, int(params.min_count)),
        )
        tail_blob = json.dumps(
            list(generated_token_ids[-tail_count:]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        event = {
            "schema": "anise.repetition-pre-stop-event.v1",
            "request_id": request_id,
            "boundary": "before_check_stop",
            "clock_id": clock_id,
            "observed_at_ns": observed_at_ns,
            "decision_at_ns": time.monotonic_ns(),
            "generated_token_count": generated_token_count,
            "current_repetition_length": current_length,
            "repeat_period_tokens": current_length,
            "repetition_count": count,
            "repetition_growth_rate": growth,
            "stop_contract": stop_contract,
            "tail_window_token_count": tail_count,
            "tail_window_end_token_index": generated_token_count,
            "tail_window_sha256": hashlib.sha256(tail_blob).hexdigest(),
            "requested_config": copy.deepcopy(config),
        }
        self._pending_pre_stop[request_id] = copy.deepcopy(event)
        if len(self._pre_stop_events) == self.capacity:
            self._pre_stop_events.popleft()
        self._pre_stop_events.append(copy.deepcopy(event))
        return event

    @staticmethod
    def _config(params: Any) -> dict[str, int]:
        config = {
            "max_pattern_size": int(params.max_pattern_size),
            "min_pattern_size": int(params.min_pattern_size),
            "min_count": int(params.min_count),
        }
        if config["max_pattern_size"] <= 0 or config["min_count"] < 2:
            raise ValueError("repetition observer received a disabled detector")
        return config

    def record_check(self, *, request_id: str, params: Any) -> None:
        request_id = str(request_id)
        config = self._config(params)
        previous = self._configs.setdefault(request_id, config)
        if previous != config:
            raise RuntimeError("repetition detector config changed in flight")
        self._checks[request_id] = self._checks.get(request_id, 0) + 1

    def record_finish(
        self,
        *,
        request_id: str,
        params: Any,
        finish_reason: Any,
        stop_reason: Any,
        generated_token_count: int,
        pre_stop_event: dict[str, Any] | None = None,
    ) -> None:
        request_id = str(request_id)
        config = self._config(params)
        if self._configs.get(request_id) != config:
            raise RuntimeError("repetition finish lacks a matching checked config")
        checks = self._checks.pop(request_id, 0)
        self._configs.pop(request_id, None)
        if checks < 1:
            raise RuntimeError("repetition detector finished without executing")
        reason_name = getattr(finish_reason, "name", None)
        normalized_reason = (
            str(reason_name).lower()
            if reason_name is not None
            else str(finish_reason).lower()
        )
        repetition_terminated = normalized_reason == "repetition"
        if repetition_terminated and stop_reason != "repetition_detected":
            raise RuntimeError("native repetition termination lacks its stop reason")
        pending_event = self._pending_pre_stop.pop(request_id, None)
        if pre_stop_event is not None and pending_event is not None and pre_stop_event != pending_event:
            raise RuntimeError("terminal receipt event differs from pending pre-stop event")
        frozen_event = pre_stop_event or pending_event
        if frozen_event is not None:
            if frozen_event.get("request_id") != request_id:
                raise RuntimeError("pre-stop event request identity differs")
            if frozen_event.get("boundary") != "before_check_stop":
                raise RuntimeError("pre-stop event boundary is invalid")
            if frozen_event.get("generated_token_count") != int(generated_token_count):
                raise RuntimeError("terminal count differs from frozen pre-stop event")
            if frozen_event.get("repeat_period_tokens") != frozen_event.get(
                "current_repetition_length"
            ):
                raise RuntimeError("pre-stop period alias differs from repetition length")
            required = {
                "tail_window_token_count",
                "tail_window_end_token_index",
                "tail_window_sha256",
                "decision_at_ns",
            }
            if not required <= set(frozen_event):
                raise RuntimeError("pre-stop tail proof is incomplete")
            if frozen_event.get("tail_window_end_token_index") != int(generated_token_count):
                raise RuntimeError("pre-stop tail proof includes future tokens")
        if len(self._receipts) == self.capacity:
            self.dropped_receipts += 1
        self.sequence += 1
        self._receipts.append(
            {
                "schema": "qbi.repetition-detection-engine-receipt.v1",
                "sequence": self.sequence,
                "timestamp_ns": time.time_ns(),
                "request_id": request_id,
                "requested_config": config,
                "effective_config": config,
                "transport_parameter_applied_proven": True,
                "detector_effective_proven": True,
                "detector_check_count": checks,
                "generated_token_count": int(generated_token_count),
                "finish_reason": normalized_reason,
                "stop_reason": stop_reason,
                "repetition_terminated": repetition_terminated,
                "pre_stop_event": copy.deepcopy(frozen_event),
            }
        )

    def state(self, after_sequence: int = 0) -> dict[str, Any]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("repetition sequence cursor must be an integer")
        if after_sequence < 0:
            raise ValueError("repetition sequence cursor must be non-negative")
        receipts = [
            copy.deepcopy(receipt)
            for receipt in self._receipts
            if int(receipt["sequence"]) > after_sequence
        ]
        oldest = int(self._receipts[0]["sequence"]) if self._receipts else 0
        return {
            "schema": "qbi.repetition-detection-engine-state.v1",
            "capacity": self.capacity,
            "oldest_available_sequence": oldest,
            "latest_sequence": self.sequence,
            "dropped_receipts": self.dropped_receipts,
            "active_request_count": len(self._checks),
            "pre_stop_event_contract": {
                "schema": "anise.repetition-pre-stop-event.v1",
                "provider_request_bound": True,
                "boundary": "before_check_stop",
                "fields": [
                    "generated_token_count",
                    "current_repetition_length",
                    "repeat_period_tokens",
                    "repetition_count",
                    "repetition_growth_rate",
                    "stop_contract",
                    "tail_window_token_count",
                    "tail_window_end_token_index",
                    "tail_window_sha256",
                    "observed_at_ns",
                    "decision_at_ns",
                    "clock_id",
                ],
            },
            "pending_pre_stop_request_ids": sorted(self._pending_pre_stop),
            "pre_stop_event_count": len(self._pre_stop_events),
            "receipts": receipts,
        }


__all__ = ["RepetitionDetectionObserver"]
