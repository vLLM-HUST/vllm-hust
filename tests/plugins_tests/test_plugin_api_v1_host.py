from pathlib import Path

from vllm.plugin_api.v1 import VllmPluginHost


def test_v1_host_builds_shared_execution_plan():
    repository = Path(__file__).resolve().parents[4]
    bundle = repository / "examples" / "plugins" / "state_affinity"
    with VllmPluginHost([bundle], features={"plugin_plan_v1"}) as host:
        plans = host.build_execution_plans({"request_class": "decode"})
        assert plans == [
            {
                "engine": "vllm_hust",
                "operator_selection": "fused_state_update_v1",
                "plan_version": 1,
                "plugin_id": "org.statecentric.state-affinity",
                "python_ipc": False,
                "request_class": "decode",
                "scheduler_policy": "state_affinity",
                "state_policy": "reuse_validated_state",
            }
        ]
