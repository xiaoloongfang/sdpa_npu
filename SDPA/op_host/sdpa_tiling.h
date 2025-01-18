/**
 * @file SDPA_tiling.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef SDPA_TILING_H
#define SDPA_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SDPATilingData)
    // data info
    TILING_DATA_FIELD_DEF(uint32_t, batch_num);
    TILING_DATA_FIELD_DEF(uint32_t, head_num);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, d);

    // device info
    // uint64_t ub_size, l1_size, l2_size, l0a_size, l0b_size, l0c_size;
    TILING_DATA_FIELD_DEF(uint64_t, ub_size);
    TILING_DATA_FIELD_DEF(uint64_t, l1_size);
    TILING_DATA_FIELD_DEF(uint64_t, l2_size);
    TILING_DATA_FIELD_DEF(uint64_t, l0a_size);
    TILING_DATA_FIELD_DEF(uint64_t, l0b_size);
    TILING_DATA_FIELD_DEF(uint64_t, l0c_size);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SDPA, SDPATilingData)
}// namespace optiling
#endif // SDPA_TILING_H
