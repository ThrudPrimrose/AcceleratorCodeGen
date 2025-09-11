#include <cstdlib>
#include "../include/GEMM.h"

int main(int argc, char **argv) {
    GEMMHandle_t handle;
    int err;
    unsigned short * __restrict__ A = (unsigned short*) calloc(262144, sizeof(unsigned short));
    unsigned short * __restrict__ B = (unsigned short*) calloc(262144, sizeof(unsigned short));
    unsigned short * __restrict__ C = (unsigned short*) calloc(262144, sizeof(unsigned short));


    handle = __dace_init_GEMM();
    __program_GEMM(handle, A, B, C);
    err = __dace_exit_GEMM(handle);

    free(A);
    free(B);
    free(C);


    return err;
}
