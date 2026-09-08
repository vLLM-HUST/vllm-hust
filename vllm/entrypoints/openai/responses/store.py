# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

T = TypeVar("T")
TDefault = TypeVar("TDefault")
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "incomplete"})


@dataclass
class _StoredValue(Generic[T]):
    value: T
    expires_at: float


class BoundedResponseStore(MutableMapping[str, T]):
    """TTL/LRU mapping that does not evict active background responses."""

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        on_evict: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._on_evict = on_evict
        self._clock = clock
        self._items: OrderedDict[str, _StoredValue[T]] = OrderedDict()

    @staticmethod
    def _is_terminal(value: T) -> bool:
        return getattr(value, "status", None) in _TERMINAL_STATUSES

    def _evict(self, key: str) -> None:
        if self._items.pop(key, None) is not None and self._on_evict is not None:
            self._on_evict(key)

    def _purge_expired(self, now: float) -> None:
        expired = [
            key
            for key, item in self._items.items()
            if item.expires_at <= now and self._is_terminal(item.value)
        ]
        for key in expired:
            self._evict(key)

    def _enforce_capacity(self) -> None:
        while len(self._items) > self.max_entries:
            terminal_key = next(
                (
                    key
                    for key, item in self._items.items()
                    if self._is_terminal(item.value)
                ),
                None,
            )
            if terminal_key is None:
                return
            self._evict(terminal_key)

    def __getitem__(self, key: str) -> T:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: T) -> None:
        now = self._clock()
        self._purge_expired(now)
        self._items[key] = _StoredValue(
            value=value,
            expires_at=now + self.ttl_seconds,
        )
        self._items.move_to_end(key)
        self._enforce_capacity()

    def __delitem__(self, key: str) -> None:
        if key not in self._items:
            raise KeyError(key)
        self._evict(key)

    def __iter__(self) -> Iterator[str]:
        self._purge_expired(self._clock())
        return iter(self._items)

    def __len__(self) -> int:
        self._purge_expired(self._clock())
        return len(self._items)

    @overload
    def get(self, key: str) -> T | None: ...

    @overload
    def get(self, key: str, default: T) -> T: ...

    @overload
    def get(self, key: str, default: TDefault) -> T | TDefault: ...

    def get(
        self, key: str, default: TDefault | None = None
    ) -> T | TDefault | None:
        now = self._clock()
        self._purge_expired(now)
        item = self._items.get(key)
        if item is None:
            return default
        if item.expires_at <= now and self._is_terminal(item.value):
            self._evict(key)
            return default
        self._items.move_to_end(key)
        return item.value
