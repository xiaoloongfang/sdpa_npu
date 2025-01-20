
#include "sdpa_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    
    SDPATilingData tiling;
    // The query shape is [batch_num, head_num, N, d]
    uint32_t batch_num = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t head_num = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t N = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    uint32_t d = context->GetInputShape(0)->GetOriginShape().GetDim(3);

    tiling.set_batch_num(batch_num);
    tiling.set_head_num(head_num);
    tiling.set_N(N);
    tiling.set_d(d);

    // Set blockdim, now that the number of participating cube cores is BLOCK_DIM.
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t core_num = ascendcPlatform.GetCoreNum();
    context->SetBlockDim(core_num);

    // Get core memory info
    uint64_t ub_size, l1_size, l2_size, l0a_size, l0b_size, l0c_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);

    tiling.set_ub_size(ub_size);
    tiling.set_l1_size(l1_size);
    tiling.set_l2_size(l2_size);
    tiling.set_l0a_size(l0a_size);
    tiling.set_l0b_size(l0b_size);
    tiling.set_l0c_size(l0c_size);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class SDPA : public OpDef {
public:
    explicit SDPA(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attn")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(SDPA);
}
