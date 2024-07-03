#include "kernel_operator.h"
#include "matmul_custom_tiling.h"

using namespace AscendC; // Otherwise auto_gen fails

constexpr size_t tileSizeK = 16;
constexpr size_t tileSizeN = 16;
constexpr size_t tileSizeM = 16;

#define GM_HALF __gm__ half *__restrict__
#define GM_FLOAT __gm__ float *__restrict__

class KernelMatMul {
public:
  __aicore__ inline KernelMatMul() {}
  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M,
                              uint32_t N, uint32_t K, uint32_t numPEInXDim,
                              uint32_t numPEInYDim) {
    this->m = M;
    this->n = N;
    this->k = K;
    size_t aSize = tileSizeM * tileSizeK; // 16 * 16; // tileSize to load, both
                                          // tiel and subtile have the sameSize
    size_t bSize = tileSizeK * tileSizeN; // 16 * 16;
    size_t cSize = tileSizeM * tileSizeN; // 16 * 16;
    this->numIterK =
        K / tileSizeK; // Amount of iterations we need to do for MatMul
    this->numIterM = (M / numPEInYDim) / tileSizeM;
    this->numIterN =
        (N / numPEInXDim) / tileSizeN; // Assuming no remainders for now
    this->myX = GetBlockIdx() % numPEInXDim;
    this->myY = GetBlockIdx() / numPEInXDim;
    this->MBegin = myY * (M / numPEInYDim);
    this->NBegin = myX * (N / numPEInXDim);

    aGm.SetGlobalBuffer((GM_HALF)a, M * K);
    bGm.SetGlobalBuffer((GM_HALF)b, K * N);
    cGm.SetGlobalBuffer((GM_FLOAT)c, M * N);

    pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
    pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
    pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
    pipe.InitBuffer(inQueueB2, 1, bSize * sizeof(half));
    pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
    pipe.InitBuffer(outQueueCO2, 1, cSize * sizeof(float));
  }

  __aicore__ inline void Process() {
    for (int tileM = 0; tileM < numIterM; tileM++) {
      for (int tileN = 0; tileN < numIterN; tileN++) {
        // We need 16x16 tile for the C Matrix we are multiplying
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        pipe_barrier(PIPE_ALL);

        for (int tileK = 0; tileK < numIterK; tileK++) {
          CopyIn(tileK, tileN, tileM);
          pipe_barrier(PIPE_ALL);
          SplitA();
          pipe_barrier(PIPE_ALL);
          SplitB();
          pipe_barrier(PIPE_ALL);
          Compute(c1Local, tileK);
          pipe_barrier(PIPE_ALL);
        }

        LocalTensor<float> c2Local = outQueueCO2.AllocTensor<float>();
        pipe_barrier(PIPE_ALL);
        Aggregate(c1Local, c2Local);
        pipe_barrier(PIPE_ALL);
        CopyOut(tileN, tileM);
        pipe_barrier(PIPE_ALL);
      }
    }
  }

private:
  __aicore__ inline void CopyAND2NZ(LocalTensor<half> &dst,
                                    GlobalTensor<half> &src, uint32_t tileK,
                                    uint32_t tileN, uint32_t tileM) {
    uint32_t srcOffset = (tileM * tileSizeM + MBegin) * k + (tileK * tileSizeK);
    uint32_t dstOffset = 0;
    // A data block is 32 bytes, (=16 halfs)
    // 16 halfs make 32 bytes, we will get 1 data block (continuously) at a
    // time, need to copy 16 blocks in total
    // The stride between 2 blocks is for A k - 16, for B n - 16 elements (in
    // multiplies of 32 bytes)
    // In dst, the distance is 0
    DataCopy(dst[dstOffset], src[srcOffset],
             {16, 1, uint16_t((k / 16) - 1), 0});
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void CopyBND2NZ(LocalTensor<half> &dst,
                                    GlobalTensor<half> &src, uint32_t tileK,
                                    uint32_t tileN, uint32_t tileM) {
    uint32_t srcOffset = (tileK * tileSizeK) * n + NBegin + tileN * tileSizeN;
    uint32_t dstOffset = 0;
    DataCopy(dst[dstOffset], src[srcOffset],
             {16, 1, uint16_t((n / 16) - 1), 0});
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void CopyIn(uint32_t tileK, uint32_t tileN,
                                uint32_t tileM) {
    LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
    pipe_barrier(PIPE_ALL);
    LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
    pipe_barrier(PIPE_ALL);
    // Transfer ND to NZ or whatever
    CopyAND2NZ(a1Local, aGm, tileK, tileN, tileM);
    pipe_barrier(PIPE_ALL);
    CopyBND2NZ(b1Local, bGm, tileK, tileN, tileM);
    pipe_barrier(PIPE_ALL);

    inQueueA1.EnQue(a1Local);
    pipe_barrier(PIPE_ALL);
    inQueueB1.EnQue(b1Local);
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void SplitA() {
    LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
    pipe_barrier(PIPE_ALL);
    LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();
    pipe_barrier(PIPE_ALL);

    LoadData2dParams loadDataParams;
    loadDataParams.repeatTimes = 1;
    loadDataParams.srcStride = 1;
    loadDataParams.ifTranspose = false;

    LoadData(a2Local, a1Local, loadDataParams);
    pipe_barrier(PIPE_ALL);
    inQueueA2.EnQue<half>(a2Local);
    pipe_barrier(PIPE_ALL);
    inQueueA1.FreeTensor(a1Local);
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void SplitB() {
    LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
    pipe_barrier(PIPE_ALL);
    LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();
    pipe_barrier(PIPE_ALL);

    LoadData2dParams loadDataParams;
    loadDataParams.repeatTimes = 1;
    loadDataParams.srcStride = 1;
    loadDataParams.ifTranspose = true;

    LoadData(b2Local, b1Local, loadDataParams);
    pipe_barrier(PIPE_ALL);
    inQueueB2.EnQue<half>(b2Local);
    pipe_barrier(PIPE_ALL);
    inQueueB1.FreeTensor(b1Local);
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void Aggregate(LocalTensor<float> &c1Local,
                                   LocalTensor<float> &c2Local) {
    outQueueCO2.EnQue<float>(c2Local);
    pipe_barrier(PIPE_ALL);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 1;
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;

    DataCopy(c2Local, c1Local, dataCopyParams, enhancedParams);
    pipe_barrier(PIPE_ALL);

    outQueueCO1.FreeTensor(c1Local);
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void Compute(const LocalTensor<float> &c1Local, int tileK) {
    LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
    pipe_barrier(PIPE_ALL);
    LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
    pipe_barrier(PIPE_ALL);
    pipe_barrier(PIPE_ALL);
    Mmad<float, half, half>(c1Local, a2Local, b2Local,
                            {uint16_t(16), uint16_t(16), uint16_t(16),
                             tileK > 0, 0, false, false, false});
    pipe_barrier(PIPE_ALL);
    inQueueB2.FreeTensor(b2Local);
    pipe_barrier(PIPE_ALL);
    inQueueA2.FreeTensor(a2Local);
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void CopyOut(int tileN, int tileM) {
    LocalTensor<float> c2Local = outQueueCO2.DeQue<float>();

    int srcOffset = 0;
    int dstOffset =
        (MBegin + (tileM * tileSizeM)) * n + (NBegin + (tileN * tileSizeN));

    // A data block is 32 bytes, (=8 floats)
    // 16 floats make 64 bytes, we will get 2 data blocks (continuously) at a
    // time, need to copy 16 blocks (len) in total
    // The stride between 2 blocks is for A k - 16, for B and C n - 16 elements
    // (in multiplies of 32 bytes it makes (n - 16)*(4/32)). In src, the
    // distance is 0
    //
    DataCopy(cGm[dstOffset], c2Local[srcOffset],
             {
                 16,
                 2,
                 0,
                 uint16_t((n / 8) - 2),
             });
    pipe_barrier(PIPE_ALL);
    outQueueCO2.FreeTensor(c2Local);
    pipe_barrier(PIPE_ALL);
  }

private:
  TPipe pipe;
  TQue<QuePosition::A1, 1> inQueueA1;
  TQue<QuePosition::A2, 1> inQueueA2;
  TQue<QuePosition::B1, 1> inQueueB1;
  TQue<QuePosition::B2, 1> inQueueB2;
  TQue<QuePosition::CO1, 1> outQueueCO1;
  TQue<QuePosition::CO2, 1> outQueueCO2;
  AscendC::GlobalTensor<half> aGm;
  AscendC::GlobalTensor<half> bGm;
  AscendC::GlobalTensor<float> cGm;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t numIterK;
  uint32_t numIterM;
  uint32_t numIterN;
  uint32_t myX;
  uint32_t myY;
  uint32_t NBegin;
  uint32_t MBegin;
};

extern "C" __global__ __aicore__ void
matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c,
              __gm__ MatMulCustomTilingData *tiling) {
  KernelMatMul op;
  op.Init(a, b, c, tiling->M, tiling->N, tiling->K, tiling->numPEInXDim,
          tiling->numPEInYDim);

  op.Process();
}

#ifndef __CCE_KT_TEST__
void matmul_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *a,
                      uint8_t *b, uint8_t *c, MatMulCustomTilingData *tiling) {
  matmul_custom<<<blockDim, l2ctrl, stream>>>(a, b, c, tiling);
}
#endif
