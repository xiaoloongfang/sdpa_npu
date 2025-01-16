/**
 * @file spda.cpp
 *
 * Copyright (C) 2022-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelSDPA {
public:
    __aicore__ inline KernelSDPA() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attn, uint32_t totalLength, uint32_t tileNum)
    {
        
    }
    __aicore__ inline void Process()
    {
        
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
       
    }
    __aicore__ inline void Compute(int32_t progress)
    {
       
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        
    }

private:
    
};

extern "C" __global__ __aicore__ void sdpa(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attn, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSDPA op;
    op.Init(query, key, value, attn, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
