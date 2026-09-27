"""Optional process-local hooks for DP load-metadata publication and adoption.

The core owns the transport and calls these factories only when a plugin
registers them. Disabled deployments keep the existing coordinator wire
format and routing behavior.
"""

from collections.abc import Callable
from typing import Any

_publisher_factory: Callable[[int], Any] | None = None
_consumer_factory: Callable[[int], Any] | None = None


def register(
    publisher_factory: Callable[[int], Any],
    consumer_factory: Callable[[int], Any],
) -> None:
    global _publisher_factory, _consumer_factory
    if _publisher_factory is not None or _consumer_factory is not None:
        if (_publisher_factory, _consumer_factory) != (
            publisher_factory,
            consumer_factory,
        ):
            raise RuntimeError("DP metadata hooks are already registered")
        return
    _publisher_factory = publisher_factory
    _consumer_factory = consumer_factory


def new_publisher(rank_count: int) -> Any | None:
    return _publisher_factory(rank_count) if _publisher_factory else None


def new_consumer(rank_count: int) -> Any | None:
    return _consumer_factory(rank_count) if _consumer_factory else None


def enabled() -> bool:
    return _publisher_factory is not None
