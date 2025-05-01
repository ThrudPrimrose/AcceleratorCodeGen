#include <cstdlib>
#include "../include/ascendc_test_1.h"

int main(int argc, char **argv) {
    ascendc_test_1Handle_t handle;
    int err;
    float * __restrict__ A = (float*) calloc(512, sizeof(float));


    handle = __dace_init_ascendc_test_1();
    __program_ascendc_test_1(handle, A);
    err = __dace_exit_ascendc_test_1(handle);

    free(A);


    return err;
}
