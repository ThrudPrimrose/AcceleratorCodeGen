#include "kernel_operator.h"


#define DIV_ROUNDUP(x,y) (((x)+(y)-1) / (y))

constexpr int64_t num_cores = 20;
constexpr int64_t vector_size = 20*32*32*32;
constexpr int64_t batch_size = 1024;
constexpr int64_t num_per_vector = 128; // vector unit: 8 datablock * 32 Bytes (per_block) / 2 Bytes (FP16)
constexpr int64_t ub_block_size = 32;

using VecBuf_t = AscendC::TBuf<AscendC::QuePosition::VECCALC>;

extern "C" __global__ __aicore__ void silu_kernel(GM_ADDR x, GM_ADDR y, int32_t num_elements)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<half, AscendC::MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);

    AscendC::GlobalTensor<half> a_gm;
    AscendC::GlobalTensor<half> b_gm;
    AscendC::GlobalTensor<half> c_gm;
    AscendC::TPipe pipe;

    VecBuf_t tbuf_a;
    pipe.InitBuffer(tbuf_a, sizeof(half) * batch_size);
    VecBuf_t tbuf_b;
    pipe.InitBuffer(tbuf_b, sizeof(half) * batch_size);
    VecBuf_t tbuf_c;
    pipe.InitBuffer(tbuf_c, sizeof(half) * batch_size);

    for (int32_t i = 0; i < vector_size; i += batch_size * num_cores){
        int32_t offset_this_core = i + batch_size * AscendC::GetBlockIdx();
        a_gm.SetGlobalBuffer((__gm__ half *)a + offset_this_core);
        b_gm.SetGlobalBuffer((__gm__ half *)b + offset_this_core);
        c_gm.SetGlobalBuffer((__gm__ half *)c + offset_this_core);

        AscendC::LocalTensor<half> local_buf_a = tbuf_a.Get<half>();
        AscendC::LocalTensor<half> local_buf_b = tbuf_b.Get<half>();
        AscendC::LocalTensor<half> local_buf_c = tbuf_c.Get<half>();

        int32_t elements_per_tile = batch_size;

        // Config for vector operation
        // See AscendC doc about repeatTimes、dataBlockStride、repeatStride
        // https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html
        uint32_t vec_repeat = DIV_ROUNDUP(elements_per_tile, num_per_vector);  // must be smaller than 255
        auto unary_params = AscendC::UnaryRepeatParams(1, 1, 8, 8);
        auto binary_params = AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8);

        // Config for copy operation
        // See AscendC doc about DataCopyParams
        // https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0102.html
        uint16_t copy_block_len = DIV_ROUNDUP(sizeof(half) * batch_size, ub_block_size);
        auto copy_params = AscendC::DataCopyParams(1, copy_block_len, 0, 0);

        int32_t x_offset = 0;
        int32_t y_offset = 0;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopy(a_buf, a_gm[x_offset], copy_params);
        a_offset += num_to_process;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopy(b_buf, b_gm[x_offset], copy_params);
        b_offset += num_to_process;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);

        AscendC::Adds<half, false>(
            c_buf, a_buf, b_buf, vec_repeat, unary_params
        );

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        AscendC::DataCopy(c_gm[y_offset], c_buf, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

        y_offset += num_to_process;
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::PipeBarrier<PIPE_ALL>();
}

void launch_silu(
    uint32_t block_dim, void *l2ctrl, void *stream,
    uint8_t *x, uint8_t *y, int32_t num_elements)
{
    silu_kernel<<<block_dim, l2ctrl, stream>>>(x, y, num_elements);
}