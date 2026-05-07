#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Unit tests for standing_high: interface and strategy.

命名约定：test_对象_断言_when_条件。注释中需写清场景、预期。
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.core.quant_service.modelslim_v1.save import AscendV1Config
from msmodelslim.core.tune_strategy.interface import EvaluateResult
from msmodelslim.core.tune_strategy.standing_high.strategy import (
    StandingHighStrategy,
    StandingHighStrategyConfig,
    get_plugin,
)
from msmodelslim.core.tune_strategy.standing_high.standing_high_interface import (
    StandingHighInterface,
)
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


class _MockModel(StandingHighInterface):
    """Mock model implementing StandingHighInterface."""

    @property
    def model_type(self):
        return "test"

    @property
    def model_path(self):
        return Path("/tmp/test")

    @property
    def trust_remote_code(self):
        return False

    def handle_dataset(self, dataset, device=DeviceType.NPU):
        return list(dataset) if dataset else []

    def load_model(self, device=DeviceType.NPU):
        m = MagicMock()
        m.to = MagicMock(return_value=None)
        return m


def _make_anti_outlier_strategies():
    """Minimal anti_outlier_strategies for config (at least one element)."""
    return [[{"type": "flex_smooth_quant"}]]


class TestStandingHighStrategyConfig:
    """StandingHighStrategyConfig 单元测试。命名：test_对象_断言_when_条件。"""

    def test_StandingHighStrategyConfig_field_match_when_valid_anti_outlier_strategies_and_default_template(self):
        """
        场景：构造配置时仅传入合法 anti_outlier_strategies，使用默认 template。
        预期：type=standing_high，anti_outlier_strategies、template.process、metadata.config_id 符合预期。
        """
        cfg = StandingHighStrategyConfig(anti_outlier_strategies=_make_anti_outlier_strategies())
        assert cfg.type == "standing_high"
        assert len(cfg.anti_outlier_strategies) >= 1
        assert len(cfg.template.process) >= 1
        assert cfg.metadata.config_id == "standing_high"

    def test_StandingHighStrategyConfig_raises_SchemaValidateError_when_anti_outlier_strategies_empty(self):
        """
        场景：anti_outlier_strategies 为空列表。
        预期：抛出 SchemaValidateError 且消息含 least one。
        """
        with pytest.raises(SchemaValidateError) as exc_info:
            StandingHighStrategyConfig(anti_outlier_strategies=[])
        assert "least one" in str(exc_info.value).lower()

    def test_StandingHighStrategyConfig_raises_SchemaValidateError_when_template_has_no_linear_quant(self):
        """
        场景：template 中不含 linear_quant 类型的 process。
        预期：抛出 SchemaValidateError 且消息含 linear_quant。
        """
        template = ModelslimV1ServiceConfig(
            process=[{"type": "flex_smooth_quant"}],
            save=[AscendV1Config(type="ascendv1_saver", part_file_size=4)],
            dataset="mix_calib.jsonl",
        )
        with pytest.raises(SchemaValidateError) as exc_info:
            StandingHighStrategyConfig(
                anti_outlier_strategies=_make_anti_outlier_strategies(),
                template=template,
            )
        assert "linear_quant" in str(exc_info.value).lower()


class TestStandingHighStrategy:
    """StandingHighStrategy 单元测试。命名：test_对象_断言_when_条件。"""

    def _make_config(self):
        return StandingHighStrategyConfig(anti_outlier_strategies=_make_anti_outlier_strategies())

    def _make_dataset_loader(self):
        loader = MagicMock()
        loader.get_dataset_by_name = MagicMock(return_value=[])
        return loader

    def test_generate_practice_raises_UnsupportedError_when_model_not_implement_StandingHighInterface(self):
        """
        场景：generate_practice 传入未实现 StandingHighInterface 的 model。
        预期：抛出 UnsupportedError 且消息含 StandingHighInterface。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)
        non_interface_model = MagicMock()
        gen = strategy.generate_practice(non_interface_model, device=DeviceType.NPU)
        with pytest.raises(UnsupportedError) as exc_info:
            next(gen)
        assert "StandingHighInterface" in str(exc_info.value)

    @patch("msmodelslim.core.tune_strategy.standing_high.strategy.LayerSelector")
    def test_generate_practice_yields_zero_practice_then_stops_when_send_is_satisfied_true(self, mock_layer_selector_cls):
        """
        场景：调用 generate_practice 后 next 取第一个 practice，再 send(is_satisfied=True)。
        预期：首项为 standing_high_ 前缀的 practice，send 后迭代器结束。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)

        mock_selector = MagicMock()
        mock_selector.run = MagicMock()
        mock_layer_selector_cls.return_value = mock_selector

        model = _MockModel()
        gen = strategy.generate_practice(model, device=DeviceType.NPU)
        practice = next(gen)
        assert practice is not None
        assert practice.spec is not None
        assert practice.metadata.config_id.startswith("standing_high_")

        result = EvaluateResult(accuracies=[], expectations=[], is_satisfied=True)
        try:
            gen.send(result)
        except StopIteration:
            pass

    def test_build_practice_config_returns_PracticeConfig_with_metadata_and_spec_when_valid_anti_outlier(self):
        """
        场景：_build_practice_config 传入合法 anti_outlier、linear_quant_exclude=[]。
        预期：返回 apiversion、metadata.config_id、spec.process、spec.dataset 与 template 一致。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)
        anti_outlier = config.anti_outlier_strategies[0]
        practice = strategy._build_practice_config(anti_outlier, linear_quant_exclude=[])
        assert practice.apiversion == "modelslim_v1"
        assert practice.metadata.config_id == "standing_high_0"
        assert len(practice.spec.process) >= 1
        assert practice.spec.dataset == config.template.dataset

    def test_build_practice_config_appends_exclude_to_linear_quant_when_linear_quant_exclude_provided(self):
        """
        场景：_build_practice_config 传入非空 linear_quant_exclude。
        预期：对应 linear_quant 的 exclude 中包含传入的项。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)
        anti_outlier = config.anti_outlier_strategies[0]
        practice = strategy._build_practice_config(
            anti_outlier,
            linear_quant_exclude=["layer.0.linear"],
        )
        assert practice.spec is not None
        linear_procs = [p for p in practice.spec.process if getattr(p, "type", None) == "linear_quant"]
        assert len(linear_procs) >= 1
        assert "layer.0.linear" in (linear_procs[0].exclude or [])

    @patch("msmodelslim.core.tune_strategy.standing_high.strategy.LayerSelector")
    def test_generate_practice_yields_multiple_practices_when_send_is_satisfied_false_then_true(self, mock_layer_selector_cls):
        """
        场景：generate_practice 后多次 send EvaluateResult（先 is_satisfied=False 再 True），直至 send(None)。
        预期：可依次取到多个 practice，最后 send(None) 触发 StopIteration。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)

        mock_selector = MagicMock()
        mock_selector.run = MagicMock()
        mock_selector.layer_groups = [["g1"], ["g2"]]
        mock_selector.select_layers_by_disable_level = MagicMock(side_effect=lambda level: [] if level == 0 else ["g2"])
        mock_layer_selector_cls.return_value = mock_selector

        model = _MockModel()
        gen = strategy.generate_practice(model, device=DeviceType.NPU)
        p1 = next(gen)
        assert p1.metadata.config_id.startswith("standing_high_")

        p2 = gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=False))
        assert p2 is not None

        p3 = gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=True))
        assert p3 is not None

        p4 = gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=True))
        assert p4 is not None

        with pytest.raises(StopIteration):
            gen.send(None)

    def test_get_plugin_returns_config_and_strategy_classes_when_called(self):
        """
        场景：调用 get_plugin()。
        预期：返回 (StandingHighStrategyConfig, StandingHighStrategy) 元组。
        """
        config_cls, strategy_cls = get_plugin()
        assert config_cls is StandingHighStrategyConfig
        assert strategy_cls is StandingHighStrategy
