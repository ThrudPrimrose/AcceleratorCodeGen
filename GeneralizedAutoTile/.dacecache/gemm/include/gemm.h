#include <dace/dace.h>
typedef void * gemmHandle_t;
extern "C" gemmHandle_t __dace_init_gemm(int K, int M, int N);
extern "C" int __dace_exit_gemm(gemmHandle_t handle);
extern "C" void __program_gemm(gemmHandle_t handle, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int K, int M, int N);
