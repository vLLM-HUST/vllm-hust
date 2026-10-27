# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``predicted_length`` field must survive the msgspec request transport.

``EngineCoreRequest`` (and the ``SamplingParams`` nested in it) crosses a
msgpack channel between the front-end and the engine core. The prediction
rides inside that struct, so a round trip through the codec must be lossless:
the value the scheduler reads is exactly the value the front-end sent.
"""

import msgspec
import pytest

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _make_request(**overrides) -> EngineCoreRequest:
    kwargs = dict(
        request_id="req-predicted",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    kwargs.update(overrides)
    return EngineCoreRequest(**kwargs)


def test_predicted_length_survives_the_msgpack_transport():
    encoded = msgspec.msgpack.encode(_make_request(predicted_length=64))
    decoded = msgspec.msgpack.decode(encoded, type=EngineCoreRequest)
    assert decoded.predicted_length == 64


def test_absent_prediction_roundtrips_as_none():
    """Requests without a prediction must not gain one on the wire."""
    encoded = msgspec.msgpack.encode(_make_request())
    decoded = msgspec.msgpack.decode(encoded, type=EngineCoreRequest)
    assert decoded.predicted_length is None
