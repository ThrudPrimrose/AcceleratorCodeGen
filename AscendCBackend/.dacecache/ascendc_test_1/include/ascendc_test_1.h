#include <dace/dace.h>
typedef void * ascendc_test_1Handle_t;
extern "C" ascendc_test_1Handle_t __dace_init_ascendc_test_1();
extern "C" int __dace_exit_ascendc_test_1(ascendc_test_1Handle_t handle);
extern "C" void __program_ascendc_test_1(ascendc_test_1Handle_t handle, float * __restrict__ A);
