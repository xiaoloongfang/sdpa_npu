#include "kernel_operator.h"

// half type, cube block: [16, 16]
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;

class KernelSDPA {
public:
    __aicore__ inline KernelSDPA()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
        mBlocks = ((m-1) / 64)+1;
        nBlocks = ((n-1) / 64)+1;
        kBlocks = ((k-1) / 64)+1;
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        aGM.SetGlobalBuffer((__gm__ half*)a);
        bGM.SetGlobalBuffer((__gm__ half*)b);
        cGM.SetGlobalBuffer((__gm__ half*)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half) / 2);
        pipe.InitBuffer(outQueueCO1, 2, cSize * sizeof(half) / 2);
        pipe.InitBuffer(outQueueCO2, 1, cSize * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();

        AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        AscendC::LocalTensor<half> c2Local = outQueueCO2.AllocTensor<half>();
        // split matrix b into 2 parts, [32, 16] and [32, 16]
        for (int i = 0; i < 2; ++i) {
            SplitB(b1Local, i);
            Compute(a2Local);
            Aggregate(c2Local, i);
        }
        inQueueB1.FreeTensor(b1Local);
        inQueueA2.FreeTensor(a2Local);
        outQueueCO2.EnQue<half>(c2Local);

        CopyOut();
    }

private:
    __aicore__ inline void CopyND2NZ(const AscendC::LocalTensor<half>& dst, const AscendC::GlobalTensor<half>& src,
        const uint16_t height, const uint16_t width)
    {
        for (int i = 0; i < width / 16; ++i) {
            int srcOffset = i * 16;
            int dstOffset = i * 16 * height;
            AscendC::DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(width / 16 - 1), 0 });
        }
    }
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        CopyND2NZ(a1Local, aGM, m, k);
        CopyND2NZ(b1Local, bGM, k, n);

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

    AscendC::GlobalTensor<half> aGM, bGM;
    AscendC::GlobalTensor<half> cGM;

    uint16_t m = 1;
    uint16_t n = 1;
    uint16_t k = 4096;

    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;

};

extern "C" __global__ __aicore__ void sdpa(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attn, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSDPA op;
    op.Init(query, key, attn);
    op.Process();
}