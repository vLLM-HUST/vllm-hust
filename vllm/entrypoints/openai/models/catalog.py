# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path
from typing import Any


def load_models_catalog(
    path: str | None,
    *,
    available_models: dict[str, int | None],
) -> list[dict[str, Any]] | None:
    """Load an optional operator-supplied model metadata catalog."""
    if path is None:
        return None
    catalog_path = Path(path).expanduser()
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to load model catalog {catalog_path}: {exc}") from exc
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("Model catalog must contain a top-level `models` array.")

    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Every model catalog entry must be an object.")
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            raise ValueError("Every model catalog entry must have a non-empty `slug`.")
        if slug in seen:
            raise ValueError(f"Duplicate model catalog slug: {slug}")
        seen.add(slug)
        if slug not in available_models:
            continue
        model_info = copy.deepcopy(item)
        max_model_len = available_models[slug]
        if max_model_len is not None:
            model_info["context_window"] = max_model_len
            model_info["max_context_window"] = max_model_len
            compact_limit = model_info.get("auto_compact_token_limit")
            if (
                not isinstance(compact_limit, int)
                or not 0 < compact_limit < max_model_len
            ):
                model_info["auto_compact_token_limit"] = min(
                    int(max_model_len * 0.75), max(1, max_model_len - 4096)
                )
        catalog.append(model_info)
    return catalog
