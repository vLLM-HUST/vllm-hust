#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Unit tests for msmodelslim.core.tune_strategy.common.config_builder.base_builder
"""
from msmodelslim.core.practice import Metadata
from msmodelslim.core.quant_service.modelslim_v1.save import AscendV1Config
from msmodelslim.core.tune_strategy.common.config_builder.base_builder import QuantizationConfigBuilder
from msmodelslim.core.tune_strategy.common.config_builder.quantization_config import (
    QuantizationConfig,
    TuningSearchSpace,
)


class ConcreteBuilder(QuantizationConfigBuilder):
    """Concrete builder for testing abstract base."""

    def build_spec_process(self, **kwargs):
        return []


class TestQuantizationConfigBuilder:
    """QuantizationConfigBuilder 基类单元测试"""

    def test_build_returns_full_QuantizationConfig_with_defaults_when_no_components_passed(self):
        """
        场景：build() 不传任何组件（使用各 build_* 默认行为）。
        预期：返回 QuantizationConfig，apiversion、metadata、spec.runner、spec.dataset、spec.save 等为默认值。
        """
        builder = ConcreteBuilder()
        config = builder.build()
        assert isinstance(config, QuantizationConfig)
        assert config.apiversion == QuantizationConfigBuilder.DEFAULT_APIVERSION
        assert config.metadata.config_id == "unknown"
        assert config.spec.runner == "auto"
        assert config.spec.dataset == QuantizationConfigBuilder.DEFAULT_DATASET
        assert config.spec.process == []
        assert len(config.spec.save) == 1
        assert isinstance(config.spec.save[0], AscendV1Config)
        assert getattr(config.spec.save[0], "type", None) == "ascendv1_saver"

    def test_build_uses_passed_metadata_when_metadata_provided(self):
        """
        场景：build() 显式传入 metadata。
        预期：返回的 config.metadata 与传入一致，不再使用 build_metadata 默认值。
        """
        builder = ConcreteBuilder()
        meta = Metadata(config_id="custom", label={"k": "v"})
        config = builder.build(metadata=meta)
        assert config.metadata.config_id == "custom"
        assert config.metadata.label == {"k": "v"}

    def test_get_tuning_search_space_returns_empty_TuningSearchSpace_when_called_without_args(self):
        """
        场景：子类未重写 get_tuning_search_space，直接调用默认实现。
        预期：返回 TuningSearchSpace 且 anti_outlier_strategies 为 None。
        """
        builder = ConcreteBuilder()
        space = builder.get_tuning_search_space()
        assert isinstance(space, TuningSearchSpace)
        assert space.anti_outlier_strategies is None
