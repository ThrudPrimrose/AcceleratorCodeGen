#include <dace/dace.h>
typedef void * GEMMHandle_t;
extern "C" GEMMHandle_t __dace_init_GEMM();
extern "C" int __dace_exit_GEMM(GEMMHandle_t handle);
extern "C" void __program_GEMM(GEMMHandle_t handle, unsigned short * __restrict__ A, unsigned short * __restrict__ B, unsigned short * __restrict__ C);
