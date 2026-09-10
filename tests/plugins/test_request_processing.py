# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.plugins import request_processing
from vllm.plugins.request_processing import (
    RequestProcessingContext,
    apply_request_processors,
    register_request_processor,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def clear_request_processors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(request_processing, "_request_processors", {})


def _context() -> RequestProcessingContext:
    return RequestProcessingContext(
        endpoint="chat",
        request_id="request-1",
        prompt_tokens=3,
        max_tokens=8,
        headers={
            "Authorization": "secret",
            "X-KV-Primary-Anchor-ID": "anchor-1",
            "X-Unrelated": "hidden",
        },
    )


def test_no_registered_processor_preserves_empty_extra_args() -> None:
    assert apply_request_processors(_context(), None) is None


def test_registered_processor_merges_owned_metadata() -> None:
    def processor(context: RequestProcessingContext) -> dict[str, object]:
        assert context.prompt_tokens == 3
        assert dict(context.headers) == {"x-kv-primary-anchor-id": "anchor-1"}
        return {"processor": context.request_id}

    register_request_processor(
        "example",
        processor,
        header_names=("X-KV-Primary-Anchor-ID",),
    )

    assert apply_request_processors(_context(), {"existing": True}) == {
        "existing": True,
        "processor": "request-1",
    }


def test_sensitive_header_access_is_rejected() -> None:
    with pytest.raises(ValueError, match="sensitive headers"):
        register_request_processor(
            "example",
            lambda _context: None,
            header_names=("authorization",),
        )


def test_header_contract_rejects_non_string_names() -> None:
    with pytest.raises(ValueError, match="must be strings"):
        register_request_processor(
            "example",
            lambda _context: None,
            header_names=(1,),  # type: ignore[arg-type]
        )


def test_processor_cannot_silently_replace_request_metadata() -> None:
    register_request_processor("example", lambda _context: {"owned": "plugin"})

    with pytest.raises(ValueError, match="conflicts with extra_args"):
        apply_request_processors(_context(), {"owned": "caller"})


def test_duplicate_processor_name_is_rejected() -> None:
    register_request_processor("example", lambda _context: None)

    with pytest.raises(ValueError, match="already registered"):
        register_request_processor("example", lambda _context: None)
