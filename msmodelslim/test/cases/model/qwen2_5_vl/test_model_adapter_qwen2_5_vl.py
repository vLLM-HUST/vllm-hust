#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError as e:
    # Importing msmodelslim triggers torch usage in logging utilities.
    raise unittest.SkipTest("torch is not installed; skip qwen2_5_vl adapter unit tests") from e

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.utils.exception import UnsupportedError


try:
    from msmodelslim.model.qwen2_5_vl.model_adapter import Qwen25VLModelAdapter
    _QWEN25_VL_IMPORT_OK = True
except Exception:
    # If transformers does not provide qwen2_5_vl modeling, importing adapter may fail.
    Qwen25VLModelAdapter = None
    _QWEN25_VL_IMPORT_OK = False


class DummyVisionConfig:
    def __init__(self, depth: int = 2):
        self.depth = depth


class DummyConfig:
    """Minimal config stub for Qwen25VLModelAdapter UT."""

    def __init__(
        self,
        num_hidden_layers: int = 3,
        output_attentions: bool = False,
        vision_depth: int = 2,
        hidden_size: int = 128,
        num_attention_heads: int = 8,
        image_token_id: int = 151655,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.output_attentions = output_attentions
        self.vision_config = DummyVisionConfig(depth=vision_depth)
        self.use_cache = True
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.image_token_id = image_token_id


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_model_type(self):
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.model_type = self.model_type
            self.assertEqual(adapter.get_model_type(), self.model_type)

    def test_get_model_pedigree(self):
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            self.assertEqual(adapter.get_model_pedigree(), "qwen25_vl")

    def test_enable_kv_cache(self):
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            model = SimpleNamespace(config=SimpleNamespace(use_cache=None))
            adapter.enable_kv_cache(model, True)
            self.assertTrue(model.config.use_cache)
            adapter.enable_kv_cache(model, False)
            self.assertFalse(model.config.use_cache)

    def test_generate_decoder_layer_yields_expected_names(self):
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.config = DummyConfig(num_hidden_layers=3)

            dummy_model = MagicMock()
            adapter._load_decoder_if_not_exist = MagicMock(side_effect=lambda _m, _n, _i: f"layer-{_i}")

            items = list(adapter.generate_decoder_layer(dummy_model))
            self.assertEqual(items, [("model.layers.0", "layer-0"), ("model.layers.1", "layer-1"), ("model.layers.2", "layer-2")])

            adapter._load_decoder_if_not_exist.assert_any_call(dummy_model, "model.layers.0", 0)
            adapter._load_decoder_if_not_exist.assert_any_call(dummy_model, "model.layers.1", 1)
            adapter._load_decoder_if_not_exist.assert_any_call(dummy_model, "model.layers.2", 2)
            self.assertEqual(adapter._load_decoder_if_not_exist.call_count, 3)

    def test_handle_dataset_requires_image_and_text(self):
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)

            # AutoProcessor.from_pretrained must be mocked to avoid resolving a real model
            # from `model_path` on environments with full transformers installed.
            with patch(
                "msmodelslim.model.qwen2_5_vl.model_adapter.AutoProcessor.from_pretrained",
                return_value=MagicMock(),
            ):
                with self.assertRaises(UnsupportedError):
                    adapter.handle_dataset([VlmCalibSample(text="hi", image=None)], device=DeviceType.CPU)

                with self.assertRaises(UnsupportedError):
                    adapter.handle_dataset([VlmCalibSample(text=None, image="a.jpg")], device=DeviceType.CPU)

    def test_handle_dataset_happy_path(self):
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)

            dataset = [VlmCalibSample(text="describe", image="a.jpg")]

            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "TEMPLATE_TEXT"
            mock_inputs = MagicMock()
            mock_processor.return_value = mock_inputs

            # Make _collect_inputs_to_device deterministic
            adapter._collect_inputs_to_device = MagicMock(return_value={"input_ids": "ok"})

            with (
                patch("msmodelslim.model.qwen2_5_vl.model_adapter.AutoProcessor.from_pretrained", return_value=mock_processor) as p_proc,
                patch("msmodelslim.model.qwen2_5_vl.model_adapter.process_vision_info", return_value=(["img"], None)) as p_vis,
                patch("msmodelslim.model.qwen2_5_vl.model_adapter.get_valid_read_path", side_effect=lambda p, *a, **k: p) as p_path,
            ):
                out = adapter.handle_dataset(dataset, device=DeviceType.CPU)

            self.assertEqual(out, [{"input_ids": "ok"}])
            p_proc.assert_called_once()
            p_vis.assert_called_once()
            p_path.assert_called()
            mock_processor.apply_chat_template.assert_called_once()
            mock_processor.assert_called_once()

            # Ensure we request expected keys for device collection
            args, kwargs = adapter._collect_inputs_to_device.call_args
            self.assertIs(args[0], mock_inputs)
            self.assertEqual(args[1], DeviceType.CPU)
            self.assertIn("keys", kwargs)
            self.assertIn("defaults", kwargs)
            self.assertIn("input_ids", kwargs["keys"])
            self.assertIn("pixel_values", kwargs["keys"])
            self.assertEqual(kwargs["defaults"].get("logits_to_keep"), 0)

    def test_init_model_calls_from_pretrained_and_restores_num_layers(self):
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.config = DummyConfig(num_hidden_layers=5, vision_depth=2)

            origin_layers = adapter.config.num_hidden_layers

            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model

            with (
                patch("msmodelslim.model.qwen2_5_vl.model_adapter.get_valid_read_path", side_effect=lambda p, *a, **k: p),
                patch("transformers.Qwen2_5_VLForConditionalGeneration") as mock_cls,
                patch.object(adapter, "_get_state_dict", return_value={}),
            ):
                mock_cls.from_pretrained.return_value = mock_model
                model = adapter.init_model(device=DeviceType.CPU)

            self.assertIs(model, mock_model)
            # restored
            self.assertEqual(adapter.config.num_hidden_layers, origin_layers)
            self.assertEqual(getattr(adapter.config, "_attn_implementation", None), "eager")
            mock_model.load_state_dict.assert_called_once()

            # called with key kwargs
            _, call_kwargs = mock_cls.from_pretrained.call_args
            self.assertEqual(call_kwargs.get("config"), adapter.config)
            self.assertEqual(call_kwargs.get("local_files_only"), True)
            self.assertEqual(call_kwargs.get("device_map"), "cpu")
            self.assertEqual(call_kwargs.get("attn_implementation"), "eager")

    def test_generate_model_visit_yields_visual_then_decoder_layers(self):
        """Test generate_model_visit yields ProcessRequest for visual first, then decoder layers."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.config = DummyConfig(num_hidden_layers=2)

            mock_visual = MagicMock()
            mock_layer0 = MagicMock()
            mock_layer1 = MagicMock()
            model = MagicMock()
            model.visual = mock_visual

            def mock_generate_decoder_layer(m):
                yield "model.layers.0", mock_layer0
                yield "model.layers.1", mock_layer1

            adapter.generate_decoder_layer = MagicMock(side_effect=mock_generate_decoder_layer)

            def mock_visit_func(m, transformer_blocks=None):
                for name, layer in transformer_blocks:
                    yield ProcessRequest(name=name, module=layer, args=(), kwargs={})

            with patch(
                "msmodelslim.model.qwen2_5_vl.model_adapter.generated_decoder_layer_visit_func",
                side_effect=mock_visit_func,
            ):
                gen = adapter.generate_model_visit(model)
                requests = list(gen)

            self.assertGreaterEqual(len(requests), 1)
            first_req = requests[0]
            self.assertIsInstance(first_req, ProcessRequest)
            self.assertEqual(first_req.name, "visual")
            self.assertIs(first_req.module, mock_visual)
            self.assertEqual(first_req.args, ())
            self.assertEqual(first_req.kwargs, {})

            decoder_requests = [r for r in requests[1:] if r.name.startswith("model.layers")]
            self.assertEqual(len(decoder_requests), 2)
            self.assertEqual(decoder_requests[0].name, "model.layers.0")
            self.assertEqual(decoder_requests[1].name, "model.layers.1")

    def test_generate_model_forward_yields_visual_then_decoder_layers(self):
        """Test generate_model_forward yields ProcessRequest for visual first, then decoder layers."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.config = DummyConfig(num_hidden_layers=2, output_attentions=False)

            mock_visual = MagicMock()
            mock_layer0 = MagicMock()
            mock_layer1 = MagicMock()
            mock_layer0.return_value = (torch.randn(1, 10, 128),)
            mock_layer1.return_value = (torch.randn(1, 10, 128),)

            model = MagicMock()
            model.visual = mock_visual
            model.config = SimpleNamespace(image_token_id=151655, output_attentions=False)
            model.model = MagicMock()
            model.model.embed_tokens = MagicMock(return_value=torch.randn(1, 10, 128))
            model.model._update_causal_mask = MagicMock(return_value=torch.ones(1, 1, 10, 10))
            model.model.rotary_emb = MagicMock(return_value=torch.randn(1, 10, 128))
            model.get_rope_index = MagicMock(
                return_value=(torch.arange(10, dtype=torch.long).unsqueeze(0), None)
            )

            def mock_generate_decoder_layer(m):
                yield "model.layers.0", mock_layer0
                yield "model.layers.1", mock_layer1

            adapter.generate_decoder_layer = MagicMock(side_effect=mock_generate_decoder_layer)

            sample = {
                "pixel_values": torch.randn(1, 3, 224, 224),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10),
            }
            sample["input_ids"][0, 0] = 151655

            gen = adapter.generate_model_forward(model, [sample])

            first_req = next(gen)
            self.assertIsInstance(first_req, ProcessRequest)
            self.assertEqual(first_req.name, "visual")
            self.assertIs(first_req.module, mock_visual)
            self.assertEqual(len(first_req.args), 2)
            self.assertEqual(first_req.args[0].shape, (1, 3, 224, 224))
            self.assertEqual(first_req.args[1].shape, (1, 3))

            image_embeds = torch.randn(1, 10, 128)
            second_req = gen.send(image_embeds)
            self.assertIsInstance(second_req, ProcessRequest)
            self.assertEqual(second_req.name, "model.layers.0")
            self.assertIs(second_req.module, mock_layer0)
            self.assertEqual(len(second_req.args), 1)
            self.assertIn("attention_mask", second_req.kwargs)
            self.assertIn("position_embeddings", second_req.kwargs)

            layer0_out = (torch.randn(1, 10, 128),)
            third_req = gen.send(layer0_out)
            self.assertIsInstance(third_req, ProcessRequest)
            self.assertEqual(third_req.name, "model.layers.1")

            with self.assertRaises(StopIteration):
                gen.send((torch.randn(1, 10, 128),))

    def test_get_ln_fuse_map_returns_empty_pre_run_and_fused_map(self):
        """Test get_ln_fuse_map returns empty pre_run dict and fused_map with correct structure."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.config = DummyConfig(num_hidden_layers=2)

            pre_run, fused_map = adapter.get_ln_fuse_map()

            self.assertIsInstance(pre_run, dict)
            self.assertEqual(len(pre_run), 0)
            self.assertIsInstance(fused_map, dict)
            self.assertIn("model.layers.0.input_layernorm", fused_map)
            self.assertIn("model.layers.0.post_attention_layernorm", fused_map)
            self.assertIn("model.norm", fused_map)
            self.assertIn("model.layers.0.self_attn.q_proj", fused_map["model.layers.0.input_layernorm"])
            self.assertEqual(fused_map["model.norm"], ["lm_head"])

    def test_get_bake_names_returns_empty_lists(self):
        """Test get_bake_names returns empty lists."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)

            pre_run_bake, bake_names = adapter.get_bake_names()

            self.assertEqual(pre_run_bake, [])
            self.assertEqual(bake_names, [])

    def test_get_rotate_map_returns_pre_run_and_rotate_pairs(self):
        """Test get_rotate_map returns pre_run and rotate_pairs with correct structure."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.config = DummyConfig(num_hidden_layers=2, hidden_size=128, num_attention_heads=8)

            pre_run_list, rot_pairs_list = adapter.get_rotate_map(block_size=8)

            self.assertIsInstance(pre_run_list, list)
            self.assertEqual(len(pre_run_list), 1)
            pre_run = pre_run_list[0]
            self.assertIsNotNone(pre_run)
            self.assertTrue(hasattr(pre_run, "left_rot") or hasattr(pre_run, "right_rot"))

            self.assertIsInstance(rot_pairs_list, list)
            self.assertGreater(len(rot_pairs_list), 0)

    def test_get_weight_map_loads_from_index_json(self):
        """Test _get_weight_map loads weight_map from model.safetensors.index.json."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)

            with tempfile.TemporaryDirectory() as tmpdir:
                adapter.model_path = tmpdir
                index_data = {
                    "weight_map": {
                        "model.layers.0.weight": "model-00001.safetensors",
                        "model.layers.1.weight": "model-00002.safetensors",
                    }
                }
                index_file = os.path.join(tmpdir, "model.safetensors.index.json")
                with open(index_file, "w") as f:
                    json.dump(index_data, f)

                with patch(
                    "msmodelslim.model.qwen2_5_vl.model_adapter.json_safe_load",
                    return_value=index_data,
                ):
                    adapter._get_weight_map.cache_clear()
                    result = adapter._get_weight_map()

                self.assertIsInstance(result, dict)
                self.assertIn("model.layers.0.weight", result)
                self.assertIn("model.layers.1.weight", result)
                self.assertEqual(result["model.layers.0.weight"], "model-00001.safetensors")
                self.assertEqual(result["model.layers.1.weight"], "model-00002.safetensors")

    def test_get_state_dict_loads_from_safetensors(self):
        """Test _get_state_dict loads weights from safetensors files."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.model_path = "."

            linear = torch.nn.Linear(4, 8)
            weight_map = {"weight": "model.safetensors", "bias": "model.safetensors"}

            mock_tensor = torch.randn(8, 4)
            mock_bias = torch.randn(8)

            with patch.object(adapter, "_get_weight_map", return_value=weight_map):
                with patch(
                    "msmodelslim.model.qwen2_5_vl.model_adapter.safe_open",
                ) as mock_safe_open:
                    def get_tensor(name):
                        if name == "weight":
                            return mock_tensor
                        return mock_bias
                    mock_f = MagicMock()
                    mock_f.get_tensor = get_tensor
                    mock_safe_open.return_value.__enter__ = MagicMock(return_value=mock_f)
                    mock_safe_open.return_value.__exit__ = MagicMock(return_value=False)

                    with patch(
                        "msmodelslim.model.qwen2_5_vl.model_adapter.get_valid_read_path",
                        side_effect=lambda p, *a, **k: p,
                    ):
                        result = adapter._get_state_dict(linear, prefix="")

                self.assertIsInstance(result, dict)
                self.assertIn("weight", result)
                self.assertIn("bias", result)
                self.assertEqual(result["weight"].shape, (8, 4))
                self.assertEqual(result["bias"].shape, (8,))

    def test_load_decoder_if_not_exist_when_layer_already_loaded(self):
        """Test _load_decoder_if_not_exist returns existing layer when already loaded."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.config = DummyConfig(num_hidden_layers=2)

            existing_layer = MagicMock()
            existing_layer.input_layernorm = MagicMock()
            existing_layer.input_layernorm.weight = MagicMock()
            existing_layer.input_layernorm.weight.device = torch.device("cpu")

            model = MagicMock()
            model.get_submodule = MagicMock(return_value=existing_layer)

            result = adapter._load_decoder_if_not_exist(model, "model.layers.0", 0)

            self.assertIs(result, existing_layer)
            model.get_submodule.assert_called_once_with("model.layers.0")

    def test_load_decoder_if_not_exist_when_layer_not_loaded(self):
        """Test _load_decoder_if_not_exist creates and loads layer when not loaded."""
        with patch("msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()):
            adapter = Qwen25VLModelAdapter(self.model_type, self.model_path, trust_remote_code=False)
            adapter.config = DummyConfig(num_hidden_layers=2)

            model = MagicMock()
            model.get_submodule = MagicMock(side_effect=AttributeError("no such module"))
            mock_module_list = []
            model.model = MagicMock()
            model.model.layers = mock_module_list

            mock_decoder = MagicMock()
            mock_decoder.eval = MagicMock(return_value=mock_decoder)

            with patch(
                "msmodelslim.model.qwen2_5_vl.model_adapter.Qwen2_5_VLDecoderLayer",
                return_value=mock_decoder,
            ):
                with patch.object(adapter, "_get_state_dict", return_value={"weight": torch.randn(1)}):
                    with patch.object(torch.nn.Linear, "reset_parameters", lambda self: None):
                        result = adapter._load_decoder_if_not_exist(model, "model.layers.0", 0)

            self.assertIs(result, mock_decoder)
            mock_decoder.load_state_dict.assert_called_once()
            self.assertIn(mock_decoder, mock_module_list)
