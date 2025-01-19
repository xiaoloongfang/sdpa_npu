/**
 * @file spda.cpp
 *
 */
#include<algorithm>

#include "kernel_operator.h"

// half type, cube block: [16, 16]
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;

class KernelSDPA {
public:
    __aicore__ inline KernelSDPA(uint32_t batch_num, uint32_t head_num, uint32_t N, uint32_t d, uint64_t Bc, uint64_t Br)
    {
        this->batch_num = batch_num;
        this->head_num = head_num;
        this->N = N;
        this->d = d;
        this->Bc = Bc;
        this->Br = Br;

        this->cache_size = uint64_t(64 * 1024)

        split_query_num = uint32_t(N-1) / Br + 1;
        split_value_num = uint32_t(N-1) / Bc + 1;

    }
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attn);
    {
        queryGM.SetGlobalBuffer((__gm__ half*)query);
        keyGM.SetGlobalBuffer((__gm__ half*)key);
        valueGM.SetGlobalBuffer((__gm__ half*)value);
        attnGM.SetGlobalBuffer((__gm__ half*)attn);

        pipe.InitBuffer(inQueueA1, 1, Br * d * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, Br * d * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, Bc * d * sizeof(half));
        pipe.InitBuffer(inQueueB2, 1, Bc * d * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, Br * Bc * sizeof(half));
        pipe.InitBuffer(outQueueCO2, 1, Br * Bc * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        
        for(uint32_t i = 0; i < batch_num; i++){
            for(uint32_t j = 0; j < head_num; j++){
                for (uint32_t c = 0; c < split_query_num; c++){
                    // 在这个里面分块计算，确保能够命中L2cache，加速！
                    CopyIn(i, j, c);
                }
                
            }
        }

        // CopyIn();
        // SplitA();

        // AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        // AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        // AscendC::LocalTensor<half> c2Local = outQueueCO2.AllocTensor<half>();
        // // split matrix b into 2 parts, [32, 16] and [32, 16]
        // for (int i = 0; i < 2; ++i) {
        //     SplitB(b1Local, i);
        //     Compute(a2Local);
        //     Aggregate(c2Local, i);
        // }
        // inQueueB1.FreeTensor(b1Local);
        // inQueueA2.FreeTensor(a2Local);
        // outQueueCO2.EnQue<half>(c2Local);

        // CopyOut();
    }

private:
    // __aicore__ inline void CopyND2NZ(const AscendC::LocalTensor<half>& dst, const AscendC::GlobalTensor<half>& src,
    //     const uint16_t height, const uint16_t width)
    // {
    //     for (int i = 0; i < width / 16; ++i) {
    //         int srcOffset = i * 16;
    //         int dstOffset = i * 16 * height;
    //         AscendC::DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(width / 16 - 1), 0 });
    //     }
    // }
    __aicore__ inline void CopyIn(uint32_t i, uint32_t j, uint32_t c)
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        // CopyND2NZ(a1Local, aGM, m, k);
        // CopyND2NZ(b1Local, bGM, k, n);

        // int srcOffset = i*head_num*N*d + j*N*d + c*d;
        // int dstOffset = 0;
        // AscendC::DataCopy(a1Local[dstOffset], queryGM[srcOffset], { d, 1, 0, 0 });
        // AscendC::DataCopy(a2Local[dstOffset], valusGM[srcOffset], { d, 1, 0, 0 });

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = Br;
        nd2nzA1Params.dValue = d;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = d;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(m) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM, nd2nzA1Params);

        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = d;
        nd2nzB1Params.dValue = Bc;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = N;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(k) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, bGM, nd2nzB1Params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA()
    {
        int srcOffset = 0;
        int dstOffset = 0;
        AscendC::LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        // transform nz to zz
        for (int i = 0; i < mBlocks; ++i) {
            AscendC::LoadData2DParams loadDataParams;
            loadDataParams.repeatTimes = kBlocks;
            loadDataParams.srcStride = mBlocks;
            loadDataParams.ifTranspose = false;

            AscendC::LoadData(a2Local[dstOffset], a1Local[srcOffset], loadDataParams);

            srcOffset += 16 * 16;
            dstOffset += k * 16;
        }

        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(const AscendC::LocalTensor<half>& b1Local, const int bSplitIdx)
    {
        AscendC::LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // transform nz to zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = kBlocks;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = true;

        AscendC::LoadData(b2Local, b1Local[bSplitIdx * bSize / 2], loadDataParams);

        inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void Compute(const AscendC::LocalTensor<half>& a2Local)
    {
        AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        AscendC::LocalTensor<half> c1Local = outQueueCO1.AllocTensor<half>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n / 2;
        mmadParams.k = k;
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

        outQueueCO1.EnQue<half>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void Aggregate(const AscendC::LocalTensor<half>& c2Local, const int bSplitIdx)
    {
        AscendC::LocalTensor<half> c1Local = outQueueCO1.DeQue<half>();

        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 2;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(c2Local[bSplitIdx * cSize / 2], c1Local, dataCopyParams, enhancedParams);

        outQueueCO1.FreeTensor(c1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> c2Local = outQueueCO2.DeQue<half>();

        // transform nz to nd
        for (int i = 0; i < nBlocks; ++i) {
            AscendC::DataCopy(cGM[i * 16], c2Local[i * m * 16], { m, 1, 0, uint16_t((nBlocks - 1) ) });
        }

        outQueueCO2.FreeTensor(c2Local);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::QuePosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::QuePosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::QuePosition::B2, 2> inQueueB2;
    // dst queue
    AscendC::TQue<AscendC::QuePosition::CO1, 2> outQueueCO1;
    AscendC::TQue<AscendC::QuePosition::CO2, 1> outQueueCO2;

    AscendC::GlobalTensor<half> queryGM, keyGM, valueGM, attnGM;

    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;

    uint32_t batch_num, head_num, N, d;

    uint64_t split_query_num, split_key_num, Bc, Br, cache_size;
    
};

extern "C" __global__ __aicore__ void sdpa(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attn, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t batch_num =  tiling_data.batch_num;
    uint32_t head_num  =  tiling_data.head_num;
    uint32_t N         =  tiling_data.N;
    uint32_t d         =  tiling_data.d;

    uint64_t ub_size   =  tiling_data.ub_size;
    uint64_t l1_size   =  tiling_data.l1_size;
    uint64_t l2_size   =  tiling_data.l2_size;
    uint64_t l0a_size  =  tiling_data.l0a_size;
    uint64_t l0b_size  =  tiling_data.l0b_size;
    uint64_t l0c_size  =  tiling_data.l0c_size;

    // 
    uint64_t Bc = min(min(l2_size / (2 * d * 4), N), 64);
    uint64_t Br = min(min(Bc, d), 64);


    KernelSDPA op(batch_num, head_num, N, d, Bc, Br);
    op.Init(query, key, value, attn, batch_num, head_num, N, d, Bc, Br);
    op.Process();
}
