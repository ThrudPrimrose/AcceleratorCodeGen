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
std::string C_PATH = std::string("C_") + std::to_string(M) + std::string("_") + std::to_string(M) + std::string("_cuda_ref.bin");

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Helper function to check cuBLAS errors
#define CHECK_CUBLAS(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS error at %s %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void matmul_cublas_fp32(float* d_A, float* d_B, float* d_C) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Set to row major mode
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    // Prepare for row-major multiplication
    // For row-major: C = A * B becomes C' = B' * A'
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // Note: cuBLAS assumes column-major order, so we transpose the operation
    // Row-major C = A * B is equivalent to column-major C' = B' * A'
    CHECK_CUBLAS(cublasSgemm(handle,
                            CUBLAS_OP_N,          // op(B)
                            CUBLAS_OP_N,          // op(A)
                            N,                    // rows of op(B) and C
                            M,                    // columns of op(A) and C
                            K,                    // cols of op(B), rows of op(A)
                            &alpha,
                            d_B,                  // B matrix
                            N,                    // leading dimension of B
                            d_A,                  // A matrix
                            K,                    // leading dimension of A
                            &beta,
                            d_C,                  // C matrix
                            N));                  // leading dimension of C

}

void read_binary(const char* filename, void* data, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data), size);
    file.close();
}

// Helper function to write binary files
void write_binary(const char* filename, const void* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(data), size);
    file.close();
}

int main() {
    // Allocate host memory for the matrices
    std::vector<half> h_A_half(M * K);
    std::vector<half> h_B_half(K * N);
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);  // Initialize C to zero

    // Read the input matrices from binary files (half precision)
    read_binary(A_PATH.c_str(), h_A_half.data(), M * K * sizeof(half));
    read_binary(B_PATH.c_str(), h_B_half.data(), K * N * sizeof(half));

    // Convert half precision matrices to single precision on the host
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __half2float(h_A_half[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __half2float(h_B_half[i]);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    // Perform matrix multiplication using cuBLAS in FP32
    matmul_cublas_fp32(d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result matrix back to host
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Write the output matrix to a binary file
    write_binary(C_PATH.c_str(), h_C.data(), M * N * sizeof(float));

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Matrix multiplication with cuBLAS completed and result saved to " << C_PATH.c_str() << std::endl;
    return 0;
}