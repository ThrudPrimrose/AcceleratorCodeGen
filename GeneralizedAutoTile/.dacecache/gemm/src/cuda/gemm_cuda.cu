
#include <cuda_runtime.h>
#include <dace/dace.h>


struct gemm_state_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(gemm_state_t *__state, int K, int M, int N);
DACE_EXPORTED int __dace_exit_cuda(gemm_state_t *__state);



int __dace_init_cuda(gemm_state_t *__state, int K, int M, int N) {
    int count;

    // Check that we are able to run cuda code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("ERROR: GPU drivers are not configured or cuda-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No cuda-capable devices found\n");
        return 2;
    }

    // Initialize cuda before we run the application
    float *dev_X;
    DACE_GPU_CHECK(cudaMalloc((void **) &dev_X, 1));
    DACE_GPU_CHECK(cudaFree(dev_X));

    

    __state->gpu_context = new dace::cuda::Context(3, 4);

    // Create cuda streams and events
    for(int i = 0; i < 3; ++i) {
        DACE_GPU_CHECK(cudaStreamCreateWithFlags(&__state->gpu_context->internal_streams[i], cudaStreamNonBlocking));
        __state->gpu_context->streams[i] = __state->gpu_context->internal_streams[i]; // Allow for externals to modify streams
    }
    for(int i = 0; i < 4; ++i) {
        DACE_GPU_CHECK(cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming));
    }

    

    return 0;
}

int __dace_exit_cuda(gemm_state_t *__state) {
    

    // Synchronize and check for CUDA errors
    int __err = static_cast<int>(__state->gpu_context->lasterror);
    if (__err == 0)
        __err = static_cast<int>(cudaDeviceSynchronize());

    // Destroy cuda streams and events
    for(int i = 0; i < 3; ++i) {
        DACE_GPU_CHECK(cudaStreamDestroy(__state->gpu_context->internal_streams[i]));
    }
    for(int i = 0; i < 4; ++i) {
        DACE_GPU_CHECK(cudaEventDestroy(__state->gpu_context->events[i]));
    }

    delete __state->gpu_context;
    return __err;
}

DACE_EXPORTED bool __dace_gpu_set_stream(gemm_state_t *__state, int streamid, gpuStream_t stream)
{
    if (streamid < 0 || streamid >= 3)
        return false;

    __state->gpu_context->streams[streamid] = stream;

    return true;
}

DACE_EXPORTED void __dace_gpu_set_all_streams(gemm_state_t *__state, gpuStream_t stream)
{
    for (int i = 0; i < 3; ++i)
        __state->gpu_context->streams[i] = stream;
}

__global__ void __launch_bounds__(64) GPU_DeviceMap_0_0_9(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int K, int M, int N) {
    {
        {
            int b_j = (256 * blockIdx.x);
            int b_i = (256 * blockIdx.y);
            {
                {
                    {
                        int64_t tmp[1024]  DACE_ALIGN(64);
                        int d_j = (32 * threadIdx.x);
                        int d_i = (32 * threadIdx.y);
                        {
                            {
                                {
                                    for (int64_t k = 0; k < K; k += 128) {
                                        __shared__ __align__(64) float B2_L1_B[32895];
                                        __shared__ __align__(64) float A2_L1_A[33533];

                                        dace::CopyND<float, 1, false, 128, 32>::template ConstDst<257, 1>::Copy(
                                        B + (((N * k) + b_j) + d_j), B2_L1_B, N, 1);

                                        dace::CopyND<float, 1, false, 32, 128>::template ConstDst<131, 1>::Copy(
                                        A + ((K * (b_i + d_i)) + k), A2_L1_A, K, 1);
                                        {
                                            #pragma unroll
                                            for (int64_t k_bl1 = 0; k_bl1 < 128; k_bl1 += 32) {
                                                float B1_L2_B[1179]  DACE_ALIGN(64);
                                                float A1_L2_A[1179]  DACE_ALIGN(64);

                                                dace::CopyND<float, 1, false, 32, 32>::template ConstDst<37, 1>::Copy(
                                                B2_L1_B + (d_j + (257 * k_bl1)), B1_L2_B, 257, 1);

                                                dace::CopyND<float, 1, false, 32, 32>::template ConstDst<37, 1>::Copy(
                                                A2_L1_A + ((131 * d_i) + k_bl1), A1_L2_A, 131, 1);
                                                {
                                                    #pragma unroll
                                                    for (int i = 0; i < 32; i += 32) {
                                                        #pragma unroll
                                                        for (int j = 0; j < 32; j += 32) {
                                                            {
                                                                #pragma unroll
                                                                for (int64_t tk = 0; tk < 32; tk += 32) {
                                                                    {
                                                                        int64_t _in_acc = tmp[((32 * i) + j)];
                                                                        float __in2 = B1_L2_B[(j + (37 * tk))];
                                                                        float __in1 = A1_L2_A[((37 * i) + tk)];
                                                                        int64_t __out;

                                                                        ///////////////////
                                                                        // Tasklet code (gemm)
                                                                        __out = GEMM(__in1, __in2, _in_acc);
                                                                        ///////////////////

                                                                        tmp[((32 * i) + j)] = __out;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                {
                                    #pragma unroll
                                    for (int64_t i = 0; i < 32; i += 1) {
                                        #pragma unroll
                                        for (int64_t j = 0; j < 32; j += 32) {
                                            {
                                                float __in1 = C[((((N * ((b_i + d_i) + i)) + b_j) + d_j) + j)];
                                                int64_t __in2 = tmp[((32 * i) + j)];
                                                float __out;

                                                ///////////////////
                                                // Tasklet code (add)
                                                __out = Add(__in1, __in2);
                                                ///////////////////

                                                C[((((N * ((b_i + d_i) + i)) + b_j) + d_j) + j)] = __out;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_GPU_DeviceMap_0_0_9(gemm_state_t *__state, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int K, int M, int N);
void __dace_runkernel_GPU_DeviceMap_0_0_9(gemm_state_t *__state, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int K, int M, int N)
{

    if ((int_ceil(N, 256)) == 0 || (int_ceil(M, 256)) == 0) {

        return;
    }

    void  *GPU_DeviceMap_0_0_9_args[] = { (void *)&A, (void *)&B, (void *)&C, (void *)&K, (void *)&M, (void *)&N };
    gpuError_t __err = cudaLaunchKernel((void*)GPU_DeviceMap_0_0_9, dim3(int_ceil(N, 256), int_ceil(M, 256), 1), dim3(8, 8, 1), GPU_DeviceMap_0_0_9_args, 0, __state->gpu_context->streams[0]);
    DACE_KERNEL_LAUNCH_CHECK(__err, "GPU_DeviceMap_0_0_9", int_ceil(N, 256), int_ceil(M, 256), 1, 8, 8, 1);
}

