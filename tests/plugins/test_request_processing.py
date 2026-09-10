# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.plugins import request_processing
from vllm.plugins.request_processing import (
    RequestProcessingContext,
    apply_request_processors,
    register_request_processor,
)


@pytest.fixture(autouse=True)
def clear_request_processors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(request_processing, "_request_processors", {})


def _context() -> RequestProcessingContext:
    return RequestProcessingContext(
        endpoint="chat",
        request_id="request-1",
        prompt_token_ids=(1, 2, 3),
        max_tokens=8,
        headers={"x-example": "value"},
    )


def test_no_registered_processor_preserves_empty_extra_args() -> None:
    assert apply_request_processors(_context(), None) is None


def test_registered_processor_merges_owned_metadata() -> None:
    def processor(context: RequestProcessingContext) -> dict[str, object]:
        assert context.prompt_tokens == 3
        return {"processor": context.request_id}

    register_request_processor("example", processor)

    assert apply_request_processors(_context(), {"existing": True}) == {
        "existing": True,
        "processor": "request-1",
    }


def test_processor_cannot_silently_replace_request_metadata() -> None:
    register_request_processor("example", lambda _context: {"owned": "plugin"})

    with pytest.raises(ValueError, match="conflicts with extra_args"):
        apply_request_processors(_context(), {"owned": "caller"})


def test_duplicate_processor_name_is_rejected() -> None:
    register_request_processor("example", lambda _context: None)

    with pytest.raises(ValueError, match="already registered"):
        register_request_processor("example", lambda _context: None)
