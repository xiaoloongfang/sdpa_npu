#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False


class TestSDPA(TestCase):

    def test_sdpa(self):
        length = [2, 48, 45, 8]
        query = torch.rand(length, device='npu', dtype=torch.float16)
        key = torch.rand(length, device='npu', dtype=torch.float16)
        value = torch.rand(length, device='npu', dtype=torch.float16)
        print(query, '\n', key, "\n", value, "\n")

        torch.npu.synchronize()
        output = torch_npu.scaled_dot_product_attention(query, key, value)
        torch.npu.synchronize()

        print(output)
        self.assertRtolEqual(output.cpu(), 
                             F.scaled_dot_product_attention(query.cpu(), key.cpu(), value.cpu()))


if __name__ == "__main__":
    run_tests()
