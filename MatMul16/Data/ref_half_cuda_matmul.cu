#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace nvcuda;

// Matrix dimensions
constexpr int M = 8192;
constexpr int N = M;
constexpr int K = M;

// Paths for the binary files
std::string A_PATH = std::string("A_") + std::to_string(M) + std::string("_") + std::to_string(M) + std::string(".bin");
std::string B_PATH = std::string("B_") + std::to_string(M) + std::string("_") + std::to_string(M) + std::string(".bin");
std::string C_PATH = std::string("C_") + std::to_string(M) + std::string("_") + std::to_string(M) + std::string("_cuda_half_ref.bin");

// CUDA error check macro
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void matmul_tensor_cores(half* A, half* B, float* C, int lda, int ldb, int ldc) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    if (warpM < M / 16 && warpN < N / 16) {
        wmma::fill_fragment(c_frag, 0.0f);

        for (int i = 0; i < K; i += 16) {
            int aRow = warpM * 16;
            int aCol = i;
            int bRow = i;
            int bCol = warpN * 16;

            // Bounds checking
            if (aRow < M && aCol < K && bRow < K && bCol < N) {
                // Load the inputs
                wmma::load_matrix_sync(a_frag, A + aRow * lda + aCol, lda);
                wmma::load_matrix_sync(b_frag, B + bRow * ldb + bCol, ldb);

                // Perform the matrix multiplication
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        wmma::store_matrix_sync(C + warpM * 16 * ldc + warpN * 16, c_frag, ldc, wmma::mem_row_major);
    }
}

// Helper functions for file I/O
void read_binary(const char* filename, void* data, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data), size);
    file.close();
}

void write_binary(const char* filename, const void* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(data), size);
    file.close();
}

void matmul_cublas_fp32(half* A, half* B, float* C, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Call cublasGemmEx for tensor core-based matmul
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k,
                 &alpha,
                 B, CUDA_R_16F, k,
                 A, CUDA_R_16F, m,
                 &beta,
                 C, CUDA_R_32F, n,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasDestroy(handle);
}

int main() {
    // Allocate host memory for the matrices
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_C_tensor_cores(M * N);
    std::vector<float> h_C_cublas(M * N);

    // Read the input matrices from binary files
    read_binary(A_PATH.c_str(), h_A.data(), M * K * sizeof(half));
    read_binary(B_PATH.c_str(), h_B.data(), K * N * sizeof(half));

    // Allocate device memory
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy matrices from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Launch custom kernel (Tensor Cores)
    dim3 threads(32, 16);
    dim3 blocks((N + 32 - 1) / 32, (M + 16 - 1) / 16);
    matmul_tensor_cores<<<blocks, threads>>>(d_A, d_B, d_C, K, N, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result matrix back to host (Tensor Cores)
    CHECK_CUDA(cudaMemcpy(h_C_tensor_cores.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Run cuBLAS-based matrix multiplication
    matmul_cublas_fp32(d_A, d_B, d_C, M, N, K);

    // Copy cuBLAS result back to host
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Write the output matrix from tensor cores to a binary file
    write_binary(C_PATH.c_str(), h_C_cublas.data(), M * N * sizeof(float));

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Numerical verification: compare the results of the two methods
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C_tensor_cores[i] - h_C_cublas[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "Maximum error between Tensor Cores and cuBLAS: " << max_error << std::endl;

    return 0;
}
