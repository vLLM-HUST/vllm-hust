# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.shared_execution import (
    SharedExecutionMetadata,
    extract_shared_execution_metadata,
)


pytestmark = pytest.mark.skip_global_cleanup


def test_extract_shared_execution_metadata_from_sampling_params():
    sampling_params = SamplingParams(
        max_tokens=16,
        extra_args={
            "shared_execution": {
                "tenant": "tenant-a",
                "canonical_prefix_key": "prefix-123",
                "retrieval_keys": ["doc-1", "doc-2"],
                "tool_calls": ["search", "search", "rerank"],
                "priority_hint": 7,
            }
        },
    )

    metadata = extract_shared_execution_metadata(sampling_params, None)

    assert metadata == SharedExecutionMetadata(
        tenant="tenant-a",
        canonical_prefix_key="prefix-123",
        retrieval_signature=("doc-1", "doc-2"),
        tool_signature=("search", "search", "rerank"),
        priority_hint=7,
    )


def test_extract_shared_execution_metadata_from_pooling_params():
    pooling_params = PoolingParams(
        extra_kwargs={
            "shared_execution": {
                "tenant": "tenant-b",
                "prefix_key": "prefix-xyz",
                "stage_id": "retrieval",
            }
        }
    )

    metadata = extract_shared_execution_metadata(None, pooling_params)

    assert metadata == SharedExecutionMetadata(
        tenant="tenant-b",
        canonical_prefix_key="prefix-xyz",
        stage_id="retrieval",
    )