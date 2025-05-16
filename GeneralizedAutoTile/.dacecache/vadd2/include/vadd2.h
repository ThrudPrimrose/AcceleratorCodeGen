#include <dace/dace.h>
typedef void * vadd2Handle_t;
extern "C" vadd2Handle_t __dace_init_vadd2(int N);
extern "C" int __dace_exit_vadd2(vadd2Handle_t handle);
extern "C" void __program_vadd2(vadd2Handle_t handle, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int N);
