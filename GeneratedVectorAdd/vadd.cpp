
#include "common.h"

#include "kernel_operator.h"

#include <cstdio>
#include <iostream>

namespace dace
{
    using float16 = half;
}

struct ascendc_test_3_state_t
{
    dace::ascendc::Context *acl_context;
};

int __dace_init_ascendc(ascendc_test_3_state_t *__state);
int __dace_exit_ascendc(ascendc_test_3_state_t *__state);

int __dace_init_ascendc(ascendc_test_3_state_t *__state)
{
    __state->acl_context = new dace::ascendc::Context(1, 1);

    DACE_ACL_CHECK(aclInit({}));
    DACE_ACL_CHECK(aclrtSetDevice(0));
    std::cout << "B0" << std::endl;
    aclrtContext *c = &__state->acl_context->aclrt_context;
    DACE_ACL_CHECK(aclrtCreateContext(c, 0));
    // Initialize acl before we run the application
    std::cout << "B0.1" << std::endl;
    float *dev_X;
    DACE_ACL_CHECK(aclrtMalloc((void **)&dev_X, 1, ACL_MEM_MALLOC_HUGE_FIRST));
    DACE_ACL_CHECK(aclrtFree(dev_X));
    std::cout << "B1" << std::endl;

    std::cout << "B3" << std::endl;

    // Create acl streams and events
    for (int i = 0; i < 1; ++i)
    {
        DACE_ACL_CHECK(aclrtCreateStream(&__state->acl_context->internal_streams[i]));
        __state->acl_context->streams[i] = __state->acl_context->internal_streams[i]; // Allow for externals to modify streams
    }
    // for(int i = 0; i < 1; ++i) {
    //     DACE_ACL_CHECK(aclrtEventCreateWithFlags(&__state->acl_context->events[i], aclrtEventDisableTiming));
    // }

    return 0;
}

int __dace_exit_ascendc(ascendc_test_3_state_t *__state)
{

    // Destroy aclrt streams and events
    for (int i = 0; i < 1; ++i)
    {
        DACE_ACL_CHECK(aclrtDestroyStream(__state->acl_context->internal_streams[i]));
    }
    // for(int i = 0; i < 1; ++i) {
    //     DACE_ACL_CHECK(aclrtDestroyEvent(__state->acl_context->events[i]));
    // }

    delete __state->acl_context;
    return 0;
}

bool __dace_acl_set_stream(ascendc_test_3_state_t *__state, int streamid, aclrtStream stream)
{
    if (streamid < 0 || streamid >= 1)
    {
        return false;
    }

    __state->acl_context->streams[streamid] = stream;

    return true;
}

void __dace_acl_set_all_streams(ascendc_test_3_state_t *__state, aclrtStream stream)
{
    for (int i = 0; i < 1; ++i)
    {
        __state->acl_context->streams[i] = stream;
    }
}

__global__ __aicore__ void __copy_map_outer_0_0_4(GM_ADDR ascend_A, GM_ADDR ascend_B)
{
    // Initialization of Global Storage
    AscendC::GlobalTensor<half> ascend_A_GM;
    AscendC::GlobalTensor<half> ascend_B_GM;
    __gm__ half *ascend_A_typed = reinterpret_cast<__gm__ half *>(ascend_A);
    __gm__ half *ascend_B_typed = reinterpret_cast<__gm__ half *>(ascend_B);

    // Initialization of Pipe and Queues
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue_frag_A;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_frag_B;
    pipe.InitBuffer(inQueue_frag_A, 1, 256 * sizeof(half));
    pipe.InitBuffer(outQueue_frag_B, 1, 256 * sizeof(half));

    {
        // Ascend Device Map
        for (long long int i = 0; i < 8192; i += 8192)
        {
            {
                {
                    // Declare frag_A
                    AscendC::LocalTensor<half> frag_A;
                    // Declare frag_B
                    AscendC::LocalTensor<half> frag_B;
                    // AiCore Group Map
                    long long int ii = (256 * AscendC::GetBlockIdx());
                    {
                        // Set Global Buffers
                        ascend_A_GM.SetGlobalBuffer(&ascend_A_typed[(i + ii)], 256);
                        ascend_B_GM.SetGlobalBuffer(&ascend_B_typed[(i + ii)], 256);
                        // Global -> VECIN: Alloc Local, DataCopy, EnQue

                        frag_A = inQueue_frag_A.AllocTensor<half>();
                        DataCopy(frag_A, ascend_A_GM, 256);
                        inQueue_frag_A.EnQue(frag_A);

                        // VECIN -> VECOUT: DeQue, Enque, Free Prev.
                        frag_A = inQueue_frag_A.DeQue<half>();
                        frag_B = outQueue_frag_B.AllocTensor<half>();
                        frag_B = frag_A;

                        outQueue_frag_B.EnQue<half>(frag_B);
                        // VECOUT -> Global: DeQue, DataCopy, Free Prev.
                        frag_B = outQueue_frag_B.DeQue<half>();
                        DataCopy(ascend_B_GM, frag_B, 256);

                        outQueue_frag_B.FreeTensor(frag_B);
                        inQueue_frag_A.FreeTensor(frag_A);
                    }
                }
            }
        }
    }
}

__global__ __aicore__ void copy_map_outer_0_0_6(GM_ADDR ascend_A, GM_ADDR ascend_B, GM_ADDR ascend_C) {
    // Initialization of Global Storage

    __gm__ dace::float16* ascend_C_typed = reinterpret_cast<__gm__ dace::float16*>(ascend_C);

    __gm__ dace::float16* ascend_A_typed = reinterpret_cast<__gm__ dace::float16*>(ascend_A);

    __gm__ dace::float16* ascend_B_typed = reinterpret_cast<__gm__ dace::float16*>(ascend_B);

    // Initialization of Pipe and Queues
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue_frag_A;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue_frag_B;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_frag_C;
    pipe.InitBuffer(inQueue_frag_B, 1, 256 * sizeof(dace::float16));
    pipe.InitBuffer(inQueue_frag_A, 1, 256 * sizeof(dace::float16));
    pipe.InitBuffer(outQueue_frag_C, 1, 256 * sizeof(dace::float16));

    {
        // Ascend Device Map
        for (long long i = 0; i < 8192; i += 1)
        {
    AscendC::GlobalTensor<dace::float16> ascend_C_GM;
    AscendC::GlobalTensor<dace::float16> ascend_A_GM;
    AscendC::GlobalTensor<dace::float16> ascend_B_GM;
            for (long long j = 0; j < 8192; j += 8192)
            {
                {
                    {
                        {
                            // Declare frag_A
                            AscendC::LocalTensor<dace::float16> frag_A;
                            // Declare frag_B
                            AscendC::LocalTensor<dace::float16> frag_B;
                            // Declare frag_C
                            AscendC::LocalTensor<dace::float16> frag_C;
                            // AiCore Group Map
                            int jj = (256 * AscendC::GetBlockIdx());
                            // AiCore Group Map
                            constexpr int ii = 0;
                            {
                                if (ii < 8192) {
                                    // Global -> VECIN: Alloc Local, DataCopy, EnQue
                                    frag_A = inQueue_frag_A.AllocTensor<dace::float16>();
                                    ascend_A_GM.SetGlobalBuffer(&ascend_A_typed[8192*i + j + jj], 256);
                                    DataCopy(frag_A, ascend_A_GM, 256);
                                    inQueue_frag_A.EnQue(frag_A);
                                    // Global -> VECIN: Alloc Local, DataCopy, EnQue
                                    frag_B = inQueue_frag_B.AllocTensor<dace::float16>();
                                    ascend_B_GM.SetGlobalBuffer(&ascend_B_typed[8192*i + j + jj], 256);
                                    DataCopy(frag_B, ascend_B_GM, 256);
                                    inQueue_frag_B.EnQue(frag_B);
                                    frag_A = inQueue_frag_A.DeQue<dace::float16>();
                                    frag_B = inQueue_frag_B.DeQue<dace::float16>();
                                    frag_C = outQueue_frag_C.AllocTensor<dace::float16>();
                                    {
                                        AscendC::LocalTensor<dace::float16>& IN_frag_A = frag_A; // Type wrapped 1
                                        AscendC::LocalTensor<dace::float16>& IN_frag_B = frag_B; // Type wrapped 1
                                        AscendC::LocalTensor<dace::float16>& OUT_frag_C = frag_C; // Type wrapped 2

                                        ///////////////////
                                        // Tasklet code (Add)
                                        Add(OUT_frag_C, IN_frag_A, IN_frag_B, 256);
                                        pipe_barrier(PIPE_ALL);
                                        ///////////////////

                                    }
                                    outQueue_frag_C.EnQue<dace::float16>(frag_C);
                                    // VECOUT -> Global: DeQue, DataCopy, Free Prev.
                                    frag_C = outQueue_frag_C.DeQue<dace::float16>();
                                    ascend_C_GM.SetGlobalBuffer(&ascend_C_typed[8192*i + j + jj], 256);
                                    DataCopy(ascend_C_GM, frag_C, 256);
                                    outQueue_frag_C.FreeTensor(frag_C);
                                }
                            }
                        }
                    }
                }
            }
            pipe_barrier(PIPE_ALL);
        }
    }
}

void __dace_runkernel_copy_map_outer_0_0_6(ascendc_test_3_state_t *__state, uint8_t* ascend_A, uint8_t* ascend_B, uint8_t* ascend_C);
void __dace_runkernel_copy_map_outer_0_0_6(ascendc_test_3_state_t *__state, uint8_t* ascend_A, uint8_t* ascend_B, uint8_t* ascend_C)
{
    #ifndef __CCE_KT_TEST__

    copy_map_outer_0_0_6<<<32, nullptr, nullptr>>>(reinterpret_cast<GM_ADDR>(ascend_A), reinterpret_cast<GM_ADDR>(ascend_B), reinterpret_cast<GM_ADDR>(ascend_C));

    DACE_ACL_CHECK(aclrtSynchronizeDevice());
    #endif
}

