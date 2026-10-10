"""Bounded prompt-free scheduler evidence for ANISE native recapture."""

from __future__ import annotations

import os
import time
import hashlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


class NativeRecaptureScopeObserver:
    """Retain only scheduling steps joined to formal recapture request IDs."""

    def __init__(self, *, enabled: bool, capacity: int, request_prefix: str) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("native recapture observer enabled must be boolean")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("native recapture observer capacity must be an integer")
        if not 128 <= capacity <= 65536:
            raise ValueError(
                "native recapture observer capacity must be in [128, 65536]"
            )
        normalized_prefix = str(request_prefix).strip()
        if not normalized_prefix:
            raise ValueError("native recapture request prefix must not be empty")
        self.enabled = enabled
        self.capacity = capacity
        self.request_prefix = normalized_prefix
        self.sequence = 0
        self._receipts: deque[dict[str, Any]] = deque(maxlen=capacity)

    @classmethod
    def from_environment(cls) -> NativeRecaptureScopeObserver:
        raw_enabled = os.environ.get("QBI_NATIVE_RECAPTURE_OBSERVER", "0")
        normalized_enabled = raw_enabled.strip().lower()
        false_values = {"0", "false", "no", "off"}
        true_values = {"1", "true", "yes", "on"}
        if normalized_enabled not in false_values | true_values:
            raise ValueError("QBI_NATIVE_RECAPTURE_OBSERVER must be a boolean flag")
        capacity_text = os.environ.get(
            "QBI_NATIVE_RECAPTURE_OBSERVER_CAPACITY", "8192"
        ).strip()
        if not capacity_text.isdecimal():
            raise ValueError(
                "QBI_NATIVE_RECAPTURE_OBSERVER_CAPACITY must be an integer"
            )
        return cls(
            enabled=normalized_enabled in true_values,
            capacity=int(capacity_text),
            request_prefix=os.environ.get(
                "QBI_NATIVE_RECAPTURE_REQUEST_PREFIX",
                "chatcmpl-anise-native-recapture-",
            ),
        )

    def record(
        self,
        *,
        num_scheduled_tokens: Mapping[str, int],
        total_num_scheduled_tokens: int,
        config_epoch: int,
        max_num_running_reqs: int,
        max_num_scheduled_tokens: int,
        request_token_state: Mapping[str, Mapping[str, int]] | None = None,
        request_prompt_token_ids: Mapping[str, Sequence[int] | None] | None = None,
    ) -> None:
        if not self.enabled:
            return
        request_ids = sorted(str(request_id) for request_id in num_scheduled_tokens)
        matched_request_ids = [
            request_id
            for request_id in request_ids
            if request_id.startswith(self.request_prefix)
        ]
        if not matched_request_ids:
            return
        positions = {}
        prompt_token_id_hashes = {}
        for request_id in request_ids:
            token_ids = (request_prompt_token_ids or {}).get(request_id)
            if token_ids is None:
                prompt_token_id_hashes[request_id] = {"status": "missing"}
            else:
                try:
                    normalized_tokens = list(token_ids)
                except TypeError:
                    normalized_tokens = []
                if not normalized_tokens or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in normalized_tokens
                ):
                    prompt_token_id_hashes[request_id] = {"status": "missing"}
                else:
                    prompt_token_id_hashes[request_id] = {
                        "status": "observed",
                        "sha256": hashlib.sha256(
                            json.dumps(
                                normalized_tokens, separators=(",", ":")
                            ).encode("utf-8")
                        ).hexdigest(),
                        "token_count": len(normalized_tokens),
                        "source": "engine_request.prompt_token_ids",
                    }
            state = (request_token_state or {}).get(request_id, {})
            computed = state.get("num_computed_tokens_before")
            prompt = state.get("num_prompt_tokens")
            scheduled = num_scheduled_tokens[request_id]
            if any(
                type(value) is not int or value < 0
                for value in (computed, prompt, scheduled)
            ):
                positions[request_id] = {"status": "missing"}
                continue
            prompt_scheduled = min(scheduled, max(0, prompt - computed))
            continuation = scheduled - prompt_scheduled
            positions[request_id] = {
                "status": "observed",
                "num_computed_tokens_before": computed,
                "num_prompt_tokens": prompt,
                "prompt_tokens_scheduled": prompt_scheduled,
                "continuation_tokens_scheduled": continuation,
                "position_phase": "mixed" if prompt_scheduled and continuation
                else "prompt" if prompt_scheduled else "continuation",
            }
        self.sequence += 1
        self._receipts.append(
            {
                "schema": "qbi.native-recapture-scheduler-scope.v1",
                "sequence": self.sequence,
                "timestamp_ns": time.time_ns(),
                "config_epoch": int(config_epoch),
                "request_ids": request_ids,
                "matched_request_ids": matched_request_ids,
                "request_count": len(request_ids),
                "matched_request_count": len(matched_request_ids),
                "formed_batch": len(request_ids) > 1,
                "num_scheduled_tokens": {
                    request_id: int(num_scheduled_tokens[request_id])
                    for request_id in request_ids
                },
                "total_num_scheduled_tokens": int(total_num_scheduled_tokens),
                "max_num_running_reqs": int(max_num_running_reqs),
                "max_num_scheduled_tokens": int(max_num_scheduled_tokens),
                "token_position_schema": "qbi.native-scheduled-token-positions.v1",
                "token_position_observation_point": "before_update_after_schedule",
                "request_token_positions": positions,
                "prompt_token_id_hashes": prompt_token_id_hashes,
                "prompt_token_observation_point": "engine_request_before_schedule_update",
                "observation_is_execution_completion": False,
            }
        )

    def state(self, after_sequence: int = 0) -> dict[str, Any]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("native recapture sequence cursor must be an integer")
        if after_sequence < 0:
            raise ValueError("native recapture sequence cursor must be non-negative")
        receipts = [
            deepcopy(receipt)
            for receipt in self._receipts
            if int(receipt["sequence"]) > after_sequence
        ]
        oldest = int(self._receipts[0]["sequence"]) if self._receipts else 0
        return {
            "schema": "qbi.native-recapture-scheduler-state.v1",
            "observer_enabled": self.enabled,
            "request_prefix": self.request_prefix,
            "capacity": self.capacity,
            "oldest_available_sequence": oldest,
            "latest_sequence": self.sequence,
            "receipts": receipts,
        }


__all__ = ["NativeRecaptureScopeObserver"]
