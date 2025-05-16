/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct gemm_state_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_GPU_DeviceMap_0_0_9(gemm_state_t *__state, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int K, int M, int N);
void __program_gemm_internal(gemm_state_t*__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int K, int M, int N)
{

    {

        __dace_runkernel_GPU_DeviceMap_0_0_9(__state, A, B, C, K, M, N);
        DACE_GPU_CHECK(cudaStreamSynchronize(__state->gpu_context->streams[0]));


    }
}

DACE_EXPORTED void __program_gemm(gemm_state_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int K, int M, int N)
{
    __program_gemm_internal(__state, A, B, C, K, M, N);
}
DACE_EXPORTED int __dace_init_cuda(gemm_state_t *__state, int K, int M, int N);
DACE_EXPORTED int __dace_exit_cuda(gemm_state_t *__state);

DACE_EXPORTED gemm_state_t *__dace_init_gemm(int K, int M, int N)
{
    int __result = 0;
    gemm_state_t *__state = new gemm_state_t;


    __result |= __dace_init_cuda(__state, K, M, N);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED int __dace_exit_gemm(gemm_state_t *__state)
{
    int __err = 0;

    int __err_cuda = __dace_exit_cuda(__state);
    if (__err_cuda) {
        __err = __err_cuda;
    }
    delete __state;
    return __err;
}
