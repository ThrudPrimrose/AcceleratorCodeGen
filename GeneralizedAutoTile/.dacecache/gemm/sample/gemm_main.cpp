#include <cstdlib>
#include "../include/gemm.h"

int main(int argc, char **argv) {
    gemmHandle_t handle;
    int err;
    int K = 42;
    int M = 42;
    int N = 42;
    float * __restrict__ A = (float*) calloc((K * M), sizeof(float));
    float * __restrict__ B = (float*) calloc((K * N), sizeof(float));
    float * __restrict__ C = (float*) calloc((M * N), sizeof(float));


    handle = __dace_init_gemm(K, M, N);
    __program_gemm(handle, A, B, C, K, M, N);
    err = __dace_exit_gemm(handle);

    free(A);
    free(B);
    free(C);


    return err;
}
