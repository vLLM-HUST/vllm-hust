"""Explicit plugin-bundle host for vLLM-HUST."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from plugin_api.v1 import PluginManager
except ImportError as exc:
    raise RuntimeError(
        "vllm.plugin_api.v1 requires the statecentric-plugin-api package"
    ) from exc


class VllmPluginHost:
    """Own plugin lifecycles and produce plans before vLLM worker execution."""

    def __init__(
        self,
        bundle_paths: Iterable[str | Path],
        *,
        features: Iterable[str],
        timeout: float = 2.0,
        allowlist: Iterable[str] | None = None,
        denylist: Iterable[str] = (),
    ) -> None:
        self._manager = PluginManager(
            bundle_paths,
            engine="vllm_hust",
            features=features,
            timeout=timeout,
            allowlist=allowlist,
            denylist=denylist,
        )

    def start(self) -> None:
        self._manager.start()

    def build_execution_plans(
        self, request_metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Build immutable plans before dispatching work to model workers."""

        return self._manager.build_plans(request_metadata)

    def health(self) -> list[dict[str, Any]]:
        return self._manager.health()

    def close(self) -> None:
        self._manager.stop()

    def __enter__(self) -> "VllmPluginHost":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
