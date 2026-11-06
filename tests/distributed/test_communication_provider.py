# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.communication_provider import (
    CommunicationProviderBase,
    load_communication_provider,
    run_collective,
)


class _TestProvider(CommunicationProviderBase):
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return input_ + 1


def test_provider_handles_selected_collective() -> None:
    native = Mock()
    provider = _TestProvider(native, "tp")
    input_ = torch.tensor([1])

    result = run_collective(provider, "all_reduce", native.all_reduce, input_)

    assert torch.equal(result, torch.tensor([2]))
    native.all_reduce.assert_not_called()


def test_provider_declines_to_native_collective() -> None:
    native = Mock()
    native.all_gather.return_value = torch.tensor([1, 1])
    provider = _TestProvider(native, "tp")
    input_ = torch.tensor([1])

    result = run_collective(provider, "all_gather", native.all_gather, input_, 0)

    assert torch.equal(result, torch.tensor([1, 1]))
    native.all_gather.assert_called_once_with(input_, 0)


def test_provider_failure_is_not_silently_hidden() -> None:
    class FailingProvider(_TestProvider):
        def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("provider failed")

    provider = FailingProvider(Mock(), "tp")

    with pytest.raises(RuntimeError, match="provider failed"):
        run_collective(provider, "all_reduce", Mock(), torch.tensor([1]))


def test_provider_is_disabled_by_default() -> None:
    assert load_communication_provider(None, Mock(), "tp") is None
