#include <cstdlib>
#include "../include/vadd2.h"

int main(int argc, char **argv) {
    vadd2Handle_t handle;
    int err;
    int N = 42;
    float * __restrict__ A = (float*) calloc(N, sizeof(float));
    float * __restrict__ B = (float*) calloc(N, sizeof(float));
    float * __restrict__ C = (float*) calloc(N, sizeof(float));


    handle = __dace_init_vadd2(N);
    __program_vadd2(handle, A, B, C, N);
    err = __dace_exit_vadd2(handle);

    free(A);
    free(B);
    free(C);


    return err;
}
