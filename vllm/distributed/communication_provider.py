# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional host contract for out-of-tree collective providers.

Providers sit above the platform's existing ``DeviceCommunicatorBase``.  They
may handle selected collectives and return ``NotImplemented`` for everything
else; the coordinator then uses the native communicator.  No provider is
loaded unless ``VLLM_HUST_COMMUNICATION_PROVIDER`` is set to its import path.
"""

from types import NotImplementedType
from typing import Any

import torch

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


class CommunicationProviderBase:
    """Base contract for an optional collective provider.

    A provider reuses the already initialized native communicator and process
    groups.  It must not create a competing default process group.
    """

    def __init__(
        self,
        device_communicator: DeviceCommunicatorBase,
        group_name: str,
    ) -> None:
        self.device_communicator = device_communicator
        self.group_name = group_name

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor | NotImplementedType:
        return NotImplemented

    def all_gather(
        self, input_: torch.Tensor, dim: int
    ) -> torch.Tensor | NotImplementedType:
        return NotImplemented

    def reduce_scatter(
        self, input_: torch.Tensor, dim: int
    ) -> torch.Tensor | NotImplementedType:
        return NotImplemented

    def broadcast(
        self, input_: torch.Tensor, src: int
    ) -> torch.Tensor | NotImplementedType:
        return NotImplemented

    def destroy(self) -> None:
        """Release provider-owned resources without destroying native groups."""


def load_communication_provider(
    qualname: str | None,
    device_communicator: DeviceCommunicatorBase | None,
    group_name: str,
) -> CommunicationProviderBase | None:
    """Load an explicitly configured provider, or keep the native path."""
    if not qualname:
        return None
    if device_communicator is None:
        raise RuntimeError(
            "VLLM_HUST_COMMUNICATION_PROVIDER requires a device communicator"
        )

    provider_cls = resolve_obj_by_qualname(qualname)
    provider = provider_cls(
        device_communicator=device_communicator,
        group_name=group_name,
    )
    if not isinstance(provider, CommunicationProviderBase):
        raise TypeError(
            f"Communication provider {qualname!r} must inherit "
            "CommunicationProviderBase"
        )
    logger.info("Loaded communication provider %s for group %s", qualname, group_name)
    return provider


def run_collective(
    provider: CommunicationProviderBase | None,
    operation: str,
    native_operation: Any,
    *args: Any,
) -> Any:
    """Run a provider collective and fall back only when it declines it."""
    if provider is not None:
        result = getattr(provider, operation)(*args)
        if result is not NotImplemented:
            return result
    return native_operation(*args)
