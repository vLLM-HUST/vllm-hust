"""Actual explicit manual policy factories for the preregistered conditional arm.

The unchanged Host controllers still own Protocol validation, decision validation,
state, failover and dispatch. This source never manufactures an invocation fact.
"""

from __future__ import annotations

import os


def activation_route():
    value = os.environ.get("ECPA_H4_CONDITIONAL_ACTIVATION")
    if value not in (None, "vanilla-vllm-entry-points", "manual-integration", "ecpa"):
        raise ValueError("unknown conditional activation route")
    return value


def manual_policy(kind, vllm_config):
    if activation_route() != "manual-integration":
        raise RuntimeError("manual factory not authorized by actual frozen arm")
    if kind == "batch":
        if (
            vllm_config.scheduler_config.batch_admission_policy
            != "vllm_hust_pipeline_microbatch.policy.PipelineMicrobatchPolicy"
        ):
            raise ValueError("manual Pipeline target differs from exact declared class")
        from vllm_hust_pipeline_microbatch.policy import PipelineMicrobatchPolicy

        return PipelineMicrobatchPolicy.from_config(
            vllm_config.scheduler_config.batch_admission_policy_config or {},
            vllm_config,
        )
    if kind == "preemption":
        if (
            vllm_config.scheduler_config.preemption_policy
            != "bidkv.adapters.vllm_hust.selector.BidkvPreemptionPolicy"
        ):
            raise ValueError("manual BidKV target differs from exact declared class")
        from bidkv.adapters.vllm_hust.selector import BidkvPreemptionPolicy

        return BidkvPreemptionPolicy.from_vllm_config(vllm_config)
    raise ValueError("unknown manual policy contract")
