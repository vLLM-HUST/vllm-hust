"""Opt-in actual scheduler-condition observations; never an activation oracle.

A missing/broken sink is fatal only in this explicitly requested future cohort.
Facts are constructed at Host call sites from actual controller/config/context;
no provider-supplied success summary or caller fact dictionary is accepted.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

GROUP = "vllm.conditional_resource_pair"
SCHEMA = "ecpa-h4-host-conditional-scheduler-fact/v1"


def enabled():
    value = os.environ.get("ECPA_H4_CONDITIONAL_EVIDENCE")
    if value not in (None, "1"):
        raise ValueError("invalid conditional evidence switch")
    return value == "1"


def evidence_mode():
    value = os.environ.get("ECPA_H4_CONDITIONAL_MODE", "qualification")
    if value not in ("qualification", "lifecycle-only", "sampled-invocation"):
        raise ValueError("unknown original conditional evidence mode")
    return value


def _json_value(value):
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(v) for v in value)
    return value


def _emit(action, payload, role):
    if not enabled():
        return
    from vllm.plugins.evidence import (
        _emit_host_event,
        allocate_controller_instance,
        allocate_invocation_sequence,
        capture_scope,
    )

    scope = capture_scope(expected_role=role)
    controller = allocate_controller_instance(scope)
    seq = allocate_invocation_sequence(scope) if action != "configured" else None
    document = dict(
        schema=SCHEMA,
        action=action,
        evidence_mode=evidence_mode(),
        payload=_json_value(payload),
        observed_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        observed_monotonic_ns=time.monotonic_ns(),
    )
    accepted = _emit_host_event(
        "resolved" if action == "configured" else "invoked",
        GROUP,
        action,
        __name__,
        detail=json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        scope=scope,
        occurrence_id=("controller:" + controller) if action == "configured" else seq,
        controller_instance_id=controller,
        observation_kind="scheduler_resolution"
        if action == "configured"
        else "scheduler_dispatch",
    )
    if accepted is not True:
        raise RuntimeError("actual conditional Host fact was not accepted by sink")


def configured(vllm_config, batch_controller, preemption_controller):
    if not enabled():
        return
    scheduler = vllm_config.scheduler_config
    parallel = vllm_config.parallel_config
    from .conditional_policy_manual import activation_route

    _emit(
        "configured",
        dict(
            activation_route=activation_route(),
            batch_controller_id=str(id(batch_controller)),
            preemption_controller_id=str(id(preemption_controller)),
            batch_policy=batch_controller.stats.policy_name,
            preemption_policy=preemption_controller.stats.policy_name,
            batch_enabled=batch_controller.enabled,
            preemption_enabled=preemption_controller.stats.enabled,
            batch_option=scheduler.batch_admission_policy,
            preemption_option=scheduler.preemption_policy,
            batch_configuration=dict(scheduler.batch_admission_policy_config or {}),
            pipeline_parallel_size=parallel.pipeline_parallel_size,
            tensor_parallel_size=parallel.tensor_parallel_size,
            max_num_seqs=scheduler.max_num_seqs,
            max_concurrent_batches=vllm_config.max_concurrent_batches,
            model=str(vllm_config.model_config.model),
        ),
        "engine-core-scheduler",
    )


def decision(controller, kind, context, policy_result, result, outcome, error=None):
    if not enabled():
        return
    if evidence_mode() == "lifecycle-only":
        return  # before selection lookup/context serialization/dispatch evidence
    if kind not in ("batch", "preemption"):
        raise ValueError("unknown actual controller")
    if os.environ.get("ECPA_H4_CONDITIONAL_SELECTION_CONTEXT"):
        from .conditional_selection_client import selected_context

        selected, control_reference = selected_context(kind, context)
        if not selected:
            return  # before context serialization/public fact emission
    else:
        control_reference = None
    payload = dict(
        controller_id=str(id(controller)),
        control_reference=control_reference,
        policy=controller.stats.policy_name,
        context=asdict(context),
        stats=asdict(controller.stats),
        policy_result=asdict(policy_result)
        if kind == "batch" and policy_result is not None
        else policy_result,
        result=asdict(result) if kind == "batch" and result is not None else result,
        outcome=outcome,
        validation_stage="original-host-validation-complete",
        error=None if error is None else type(error).__name__,
    )
    _emit(kind, payload, "engine-core-scheduler")


def mapping(request):
    if enabled():
        if evidence_mode() == "lifecycle-only":
            return  # no API mapping publication or selection IPC in lifecycle mode
        if os.environ.get("ECPA_H4_CONDITIONAL_SELECTION_CONTEXT"):
            from .conditional_selection_client import publish_mapping

            selected, control_reference = publish_mapping(request)
            if not selected:
                return  # authenticated unselected original RID, no sampled fact
        else:
            control_reference = None
        if not request.external_req_id or not request.request_id:
            raise ValueError("actual API external/runtime IDs absent")
        _emit(
            "mapping",
            dict(
                control_reference=control_reference,
                external_request_id=request.external_req_id,
                runtime_request_id=request.request_id,
            ),
            "api-server",
        )
