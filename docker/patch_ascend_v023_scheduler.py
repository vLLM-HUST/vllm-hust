"""Apply the minimal scheduler-policy seam to the pinned Ascend v0.23 base."""

import sys
from pathlib import Path


SCHEDULER = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py")
)


def replace_once(source: str, old: str, new: str) -> str:
    """Replace one pinned-base fragment and fail closed on drift."""
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one scheduler fragment: {old[:80]!r}")
    return source.replace(old, new, 1)


source = SCHEDULER.read_text()

source = replace_once(
    source,
    "from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector\n"
    "from vllm.v1.core.sched.interface import PauseState, SchedulerInterface\n",
    "from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector\n"
    "from vllm.v1.core.sched.victim_selector import (\n"
    "    get_victim_selector,\n"
    "    infer_kv_utilization_from_scheduler,\n"
    ")\n"
    "from vllm.v1.core.sched.interface import PauseState, SchedulerInterface\n",
)

source = replace_once(
    source,
    "        self.running: list[Request] = []\n\n"
    "        # The request IDs that are finished in between the previous and the\n",
    "        self.running: list[Request] = []\n\n"
    "        self.victim_selector = get_victim_selector(self.vllm_config)\n\n"
    "        # The request IDs that are finished in between the previous and the\n",
)

old_preemption = """                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()
"""

new_preemption = """                    # The request cannot be scheduled.
                    preempted_req = self.victim_selector.pick_victim(
                        self.running,
                        self.policy,
                        kv_utilization=infer_kv_utilization_from_scheduler(self),
                        now_s=scheduled_timestamp,
                    )
                    self.running.remove(preempted_req)
                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens.pop(preempted_req_id)
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                            preempted_req_id, None
                        )
                        if preempted_encoder_inputs:
                            # Restore encoder compute budget if the preempted
                            # request had encoder inputs scheduled in this step.
                            num_embeds_to_restore = sum(
                                preempted_req.get_num_encoder_embeds(i)
                                for i in preempted_encoder_inputs
                            )
                            encoder_compute_budget += num_embeds_to_restore
                        req_index -= 1
"""
source = replace_once(source, old_preemption, new_preemption)

source = replace_once(
    source,
    "        with record_function_or_nullcontext(\"schedule: update_after_schedule\"):\n"
    "            self._update_after_schedule(scheduler_output)\n"
    "        return scheduler_output\n",
    "        with record_function_or_nullcontext(\"schedule: update_after_schedule\"):\n"
    "            self._update_after_schedule(scheduler_output)\n\n"
    "        if preempted_reqs:\n"
    "            self.victim_selector.emit_observability_log(\n"
    "                logger, self.__class__.__name__\n"
    "            )\n\n"
    "        return scheduler_output\n",
)

SCHEDULER.write_text(source)
