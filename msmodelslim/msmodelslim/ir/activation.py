#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import torch
import torch.nn as nn

from msmodelslim.ir.api import fake_quantize
from msmodelslim.ir.qal import QABCRegistry, QScheme, QScope, QDType, QParam, QStorage
from msmodelslim.utils.logging import logger_setter
from .auto import AutoFakeQuantActivation
from .const import int8_per_head_sym


@QABCRegistry.multi_register(
    dispatch_key=[
        int8_per_head_sym,
    ],
    abc_type=AutoFakeQuantActivation
)
@logger_setter()
class FakeQuantActivationPerHead(AutoFakeQuantActivation):
    """对称 per-head 伪量化/反量化。

    输入形状: (batch_size, num_head, seq_len, head_dim)，按第 1 维做 per-head。
    仅需 ext['scale']，忽略 offset。
    """

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

        scale = x_q_param.ext.get("scale")
        if scale is None:
            raise ValueError("`scale` is needed in ext but is missing for FakeQuantActivationPerHead")

        self.input_scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将 head 维（固定为索引 1）移动到最后一维，按每通道量化最后一维
        x_perm = x.movedim(1, -1)
        last_dim = x_perm.shape[-1]
        x_2d = x_perm.reshape(-1, last_dim)

        per_channel_scheme = QScheme(scope=QScope.PER_CHANNEL, dtype=self.x_q_scheme.dtype,
                                     symmetric=True)
        ext = {"scale": self.input_scale.data}

        x_q_param = QParam(scheme=per_channel_scheme, ext=ext)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_2d), x_q_param)

        x_out = x_q_dq.value.reshape(x_perm.shape)
        x_out = x_out.movedim(-1, 1)
        return x_out
