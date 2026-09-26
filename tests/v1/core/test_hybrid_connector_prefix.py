# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from unittest.mock import Mock

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator


def test_connector_without_state_restore_uses_common_prefix():
    scheduler = create_scheduler(
        model=os.environ.get("VLLM_TEST_MODEL", "facebook/opt-125m"),
        enable_prefix_caching=True,
    )
    scheduler.has_mamba_layers = True
    coordinator = Mock(
        spec=HybridKVCacheCoordinator,
        wraps=scheduler.kv_cache_manager.coordinator,
    )
    coordinator.num_uncached_common_prefix_tokens = 0
    coordinator.find_longest_cache_hit_per_group = Mock(
        side_effect=AssertionError(
            "Attention-only cache hits require restoration of missing recurrent state"
        )
    )
    scheduler.kv_cache_manager.coordinator = coordinator
    connector = Mock(spec=KVConnectorBase_V1)
    connector.supports_independent_hybrid_cache_hits = (
        KVConnectorBase_V1.supports_independent_hybrid_cache_hits.fget(connector)
    )
    connector.get_num_new_matched_tokens.return_value = (0, False)
    scheduler.connector = connector
    request = create_requests(num_requests=1, num_tokens=32)[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 32
    coordinator.find_longest_cache_hit.assert_called_once()
    coordinator.find_longest_cache_hit_per_group.assert_not_called()
    connector.get_num_new_matched_tokens.assert_called_once_with(request, 0)
