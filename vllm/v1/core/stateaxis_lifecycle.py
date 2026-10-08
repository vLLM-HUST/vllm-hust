# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded observation of the existing scheduler's lifecycle.

This module does not allocate, synchronize, schedule, or authorize resources.
KV reference return is not device-memory deallocation. Events are diagnostic
evidence; dropped events explicitly invalidate a complete-trace claim.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestEpoch:
    request_id: str
    generation: int
    execution_epoch: int


class LifecycleObserver:
    """EngineCore-thread-owned observer; no callbacks or hot-path I/O."""

    def __init__(self, max_events: int = 4096) -> None:
        if type(max_events) is not int or not 1 <= max_events <= 65536:
            raise ValueError("stateaxis max_events must be an integer in [1, 65536]")
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.dropped_events = 0
        self._sequence = 0
        self._generation = 0
        self._step = 0
        self._requests: dict[str, tuple[object, RequestEpoch]] = {}
        self._steps: dict[int, tuple[object, int, tuple[RequestEpoch, ...]]] = {}
        self._completed: set[int] = set()
        self._submitted: set[int] = set()
        self._deferred: dict[int, tuple[object, RequestEpoch, int]] = {}

    @classmethod
    def from_additional_config(cls, config: object) -> "LifecycleObserver | None":
        # vLLM also accepts opaque SupportsHash objects, without a mapping API.
        return cls.from_config(
            config.get("stateaxis_lifecycle") if isinstance(config, dict) else None
        )

    @classmethod
    def from_config(cls, config: object) -> "LifecycleObserver | None":
        if config is None:
            return None
        if not isinstance(config, dict) or set(config) - {"mode", "max_events"}:
            raise ValueError("invalid stateaxis_lifecycle configuration")
        mode = config.get("mode", "off")
        if mode == "off":
            return None
        if mode != "observe":
            raise ValueError("stateaxis lifecycle supports only off/observe")
        return cls(config.get("max_events", 4096))

    def record(self, kind: str, **data: Any) -> None:
        if len(self.events) == self.events.maxlen:
            self.dropped_events += 1
        self._sequence += 1
        self.events.append({"sequence": self._sequence, "kind": kind, **data})

    def admitted(self, request: Any) -> None:
        if request.request_id in self._requests:
            raise RuntimeError("duplicate lifecycle admission")
        self._generation += 1
        token = RequestEpoch(request.request_id, self._generation, 0)
        self._requests[request.request_id] = (request, token)
        self.record("request_admitted", token=token)

    def token(self, request: Any) -> RequestEpoch:
        owner, token = self._requests[request.request_id]
        if owner is not request:
            raise RuntimeError("request generation identity mismatch")
        return token

    def preempted(self, request: Any) -> None:
        old = self.token(request)
        new = RequestEpoch(old.request_id, old.generation, old.execution_epoch + 1)
        self._requests[request.request_id] = (request, new)
        self.record("request_preempted", previous=old, token=new)

    def retired(self, request: Any) -> None:
        token = self.token(request)
        del self._requests[request.request_id]
        self.record("request_detached", token=token)

    def scheduled(self, output: Any, requests: dict[str, Any]) -> None:
        key = id(output)
        if key in self._steps:
            raise RuntimeError("scheduler output already observed")
        tokens = tuple(self.token(requests[rid]) for rid in output.num_scheduled_tokens)
        self._step += 1
        # Retain the actual output object to prevent identity reuse until commit.
        self._steps[key] = (output, self._step, tokens)
        self.record("step_scheduled", step=self._step, tokens=tokens)

    def execution_completed(self, output: Any) -> None:
        key = id(output)
        owner, step, tokens = self._steps[key]
        if owner is not output or key in self._completed or key not in self._submitted:
            raise RuntimeError("duplicate or mismatched completion")
        self._completed.add(key)
        current = tuple(
            self._requests.get(token.request_id, (None, None))[1] == token
            for token in tokens
        )
        self.record(
            "execution_completed",
            step=step,
            tokens=tokens,
            epoch_current=current,
        )

    def execution_submitted(self, output: Any) -> None:
        key = id(output)
        owner, step, _ = self._steps[key]
        if owner is not output or key in self._submitted:
            raise RuntimeError("duplicate or mismatched execution submission")
        self._submitted.add(key)
        self.record("execution_submitted", step=step)

    def committed(self, output: Any) -> None:
        key = id(output)
        if key not in self._completed:
            raise RuntimeError("scheduler commit before execution completion")
        _, step, _ = self._steps.pop(key)
        self._completed.remove(key)
        self._submitted.remove(key)
        self.record("scheduler_update_completed", step=step)

    def deferred(self, request: Any, blocks: list[Any], fence: int) -> None:
        token = self.token(request)
        key = id(blocks)
        if key in self._deferred:
            raise RuntimeError("duplicate deferred block group")
        self._deferred[key] = (blocks, token, fence)
        self.record(
            "kv_refs_deferred",
            token=token,
            fence=fence,
            blocks=tuple(block.block_id for block in blocks),
        )

    def deferred_returned(self, blocks: list[Any]) -> None:
        owner, token, fence = self._deferred.pop(id(blocks))
        if owner is not blocks:
            raise RuntimeError("deferred block identity mismatch")
        self.record(
            "kv_refs_returned",
            token=token,
            fence=fence,
            blocks=tuple((block.block_id, block.ref_cnt) for block in blocks),
        )

    def snapshot(self) -> dict[str, Any]:
        """Copy evidence outside the token path; unresolved work stays visible."""
        from dataclasses import asdict

        def encode(value: Any) -> Any:
            if isinstance(value, RequestEpoch):
                return asdict(value)
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            return value

        return {
            "schema": "stateaxis-vllm-lifecycle-observation-v1",
            "governance_enforced": False,
            "dropped_events": self.dropped_events,
            "trace_lossless": self.dropped_events == 0,
            "observed_lifetimes_closed": not (
                self._requests or self._steps or self._deferred
            ),
            "active_requests": len(self._requests),
            "uncommitted_steps": len(self._steps),
            "completed_uncommitted_steps": len(self._completed),
            "submitted_uncommitted_steps": len(self._submitted),
            "deferred_block_groups": len(self._deferred),
            "events": encode(list(self.events)),
        }
