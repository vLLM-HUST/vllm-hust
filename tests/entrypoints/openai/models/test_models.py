# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.models.catalog import load_models_catalog

# any model with a chat template should work here
MODEL_NAME = "Qwen/Qwen3-0.6B"
# technically this needs Mistral-7B-v0.1 as base, but we're not testing
# generation quality here


@pytest.fixture(scope="module")
def server(qwen3_lora_files):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--lora-modules",
        f"qwen3-lora={qwen3_lora_files}",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_check_models(client: openai.AsyncOpenAI, qwen3_lora_files):
    models = await client.models.list()
    models = models.data
    served_model = models[0]
    lora_models = models[1:]
    assert served_model.id == MODEL_NAME
    assert served_model.root == MODEL_NAME
    assert all(lora_model.root == qwen3_lora_files for lora_model in lora_models)
    assert lora_models[0].id == "qwen3-lora"


def test_operator_catalog_preserves_openai_models_and_uses_live_context(
    tmp_path,
) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": MODEL_NAME,
                        "display_name": "Qwen test",
                        "context_window": 1,
                        "auto_compact_token_limit": 999999,
                    },
                    {"slug": "not-served", "display_name": "Hidden"},
                ]
            }
        )
    )

    catalog = load_models_catalog(str(path), available_models={MODEL_NAME: 8192})

    assert catalog is not None
    assert [item["slug"] for item in catalog] == [MODEL_NAME]
    assert catalog[0]["context_window"] == 8192
    assert catalog[0]["max_context_window"] == 8192
    assert catalog[0]["auto_compact_token_limit"] == 4096
