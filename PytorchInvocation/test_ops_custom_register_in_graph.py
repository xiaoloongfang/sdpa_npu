#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

from typing import Any
import torch
from torch.library import impl
import torch_npu
import torchair
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor
from torch_npu.meta._meta_registrations import m


@impl(m, "npu_sdpa")
def npu_sdpa_meta(x, y):
    return torch.empty_like(x)


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.npu.npu_sdpa.default)
def convert_npu_sdpa(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "SDPA",
        inputs={
            "x": x,
            "y": y,
        },
        outputs=['z']
    )


class TestTorchCompileSDPA(TestCase):

    def test_sdpa(self):
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        length = [2, 48, 45, 8]
        query = torch.rand(length, device='npu', dtype=torch.float16)
        key = torch.rand(length, device='npu', dtype=torch.float16)
        value = torch.rand(length, device='npu', dtype=torch.float16)
        print(query, '\n', key, "\n", value, "\n")
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q, k, v):
                return torch_npu.scaled_dot_product_attention(q, k, v)
        mod = torch.compile(Module().npu(), backend=npu_backend)
        output = mod(query, key, value)
        print(output)
        self.assertRtolEqual(output.cpu(), 
                            F.scaled_dot_product_attention(
                                query.cpu(), key.cpu(), value.cpu()))


if __name__ == "__main__":
    run_tests()
