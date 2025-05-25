#include "matmul_custom_tiling.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h> // For PATH_MAX
#include <sstream>
#include <stdlib.h> // For realpath
#include <string.h>

#include "../Util/util.h"

constexpr uint32_t dim = 64;

#define CHECK_ACL(x)                                                    \
  do                                                                    \
  {                                                                     \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE)                                        \
    {                                                                   \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);

#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
#include "acl/ops/acl_cblas.h"

extern void matmul_custom_do(uint32_t numPE, void *l2ctrl, void *stream,
                             uint8_t *a, uint8_t *b, uint8_t *c,
                             MatMulCustomTilingData *data);
#endif

int32_t main(int32_t argc, char *argv[])
{
  size_t inputByteSize = dim * dim * sizeof(uint16_t);
  size_t outputByteSize = dim * dim * sizeof(float);

#ifndef __CCE_KT_TEST__
  // Check this
  // see
  // https://www.hiascend.com/document/detail/en/canncommercial/700/inferapplicationdev/aclcppdevg/aclcppdevg_0066.html
  const char *relative_path = __FILE__;

  // Buffer to hold the absolute path
  char absolute_path[PATH_MAX];
  char *last_slash = nullptr;
  // Convert the relative path to an absolute path
  if (realpath(relative_path, absolute_path) != NULL)
  {
    std::cout << "Absolute path of the compiled file: " << absolute_path
              << std::endl;
    last_slash = strrchr(absolute_path, '/');

    if (last_slash != NULL)
    {
      // Terminate the string at the last slash to get the directory path
      *last_slash = '\0';
    }
  }
  else
  {
    std::cerr << "Error resolving absolute path." << std::endl;
  }
  std::string absolute_path_str(absolute_path);
  std::string A_path = absolute_path_str + "/Data/A.bin";
  std::string B_path = absolute_path_str + "/Data/B.bin";
  std::string C_ref_path = absolute_path_str + "/Data/C_ref.bin";
  std::string C_half_ref_path = absolute_path_str + "/Data/C_half_ref.bin";
  std::string C_path = absolute_path_str + "/Data/C.txt";
  std::string C_aclblas_path = absolute_path_str + "/Data/C_aclblas.txt";

  std::string gemm_path = absolute_path_str + "/op_models";
  std::cout << "Op dir: " << gemm_path << std::endl;
  std::string acl_path = absolute_path_str + "/op_models/acl.json";
  CHECK_ACL(aclInit(acl_path.c_str()));
  CHECK_ACL(aclopSetModelDir(gemm_path.c_str()));
  aclrtRunMode runMode = ACL_DEVICE;
  aclrtContext context;
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  MatMulCustomTilingData *tiling;
  MatMulCustomTilingData *tilingDevice;
  uint8_t *xHost, *yHost;
  uint8_t *zHost;
  uint8_t *zHost2;
  float *cRefHost = new float[dim * dim];
  aclFloat16 *cHalfRefHost = new aclFloat16[dim * dim];
  // void *workSpaceHost;
  uint8_t *xDevice, *yDevice;
  uint8_t *zDevice;
  float *alpha;
  float *beta;
  float hostalpha = 1.0f;
  float hostbeta = 0.0f;
  // void *workSpaceDevice;
  CHECK_ACL(
      aclrtMallocHost((void **)(&tiling), sizeof(MatMulCustomTilingData)));
  CHECK_ACL(aclrtMalloc((void **)(&tilingDevice),
                        sizeof(MatMulCustomTilingData),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  tiling->M = dim;
  tiling->N = dim;
  tiling->K = dim;
  tiling->numPEInXDim = 4;
  tiling->numPEInYDim = 4;
  uint32_t numPE = 16;
  CHECK_ACL(aclrtMemcpy(tilingDevice, sizeof(MatMulCustomTilingData), tiling,
                        sizeof(MatMulCustomTilingData),
                        ACL_MEMCPY_HOST_TO_DEVICE));

  CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
  CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
  CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
  CHECK_ACL(aclrtMallocHost((void **)(&zHost2), outputByteSize));
  CHECK_ACL(
      aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(
      aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(
      aclrtMalloc((void **)&alpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(
      aclrtMalloc((void **)&beta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));

  readMatrixFromBinaryFile(xHost, A_path, dim, dim, sizeof(aclFloat16));
  readMatrixFromBinaryFile(yHost, B_path, dim, dim, sizeof(aclFloat16));
  readMatrixFromBinaryFile(cRefHost, C_ref_path, dim, dim, sizeof(float));
  readMatrixFromBinaryFile(cHalfRefHost, C_half_ref_path, dim, dim, sizeof(aclFloat16));

  bool all_zero = true;
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      aclFloat16 f = xHost[i * dim + j];
      if (aclFloat16ToFloat(f) != 0.f)
      {
        all_zero = false;
        break;
      }
    }
  }
  assert(!all_zero);

  all_zero = true;
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      aclFloat16 f = yHost[i * dim + j];
      if (aclFloat16ToFloat(f) != 0.f)
      {
        all_zero = false;
        break;
      }
    }
  }

  CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(alpha, sizeof(float), &hostalpha, sizeof(float),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(beta, sizeof(float), &hostbeta, sizeof(float),
                        ACL_MEMCPY_HOST_TO_DEVICE));

  std::cout << "start self kernel" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  {
    // CHECK_ACL(ACLRT_LAUNCH_KERNEL(matmul_custom)(numPE, stream, xDevice,
    //                                              yDevice, zDevice, tiling));
    matmul_custom_do(numPE, nullptr, stream, (uint8_t *)xDevice,
                     (uint8_t *)yDevice, (uint8_t *)zDevice, tilingDevice);
  }

  CHECK_ACL(aclrtSynchronizeStream(stream));
  CHECK_ACL(aclrtSynchronizeDevice());
  auto end = std::chrono::high_resolution_clock::now();
  double duration =
      ((double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
           .count()) /
      1e12;
  std::cout << "end self kernel" << std::endl;
  std::cout << "Time self kernel: " << duration << " seconds" << std::endl;
  double num_ops = double(dim) * double(dim) * double(dim) * 2.0;
  double peak_flops = 2.2 * 1e12;
  std::cout << "#FLOPS: " << num_ops << ", Peak FLOPS/s: " << peak_flops << ", "
            << double(num_ops / peak_flops)
            << "(theoretical upper limit in seconds)" << std::endl;

  CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  CHECK_ACL(aclrtMemset(zDevice, outputByteSize, 0, outputByteSize));

  std::cout << "start aclblas kernel" << std::endl;
  auto start_acl = std::chrono::high_resolution_clock::now();

  // Check reserved parameters etc
  // https://www.hiascend.com/en/document/detail/zh/canncommercial/80RC1/apiref/appdevgapi/aclcppdevg_03_0222.html
  // setting lda, ldb, ldc != -1 resutls with err 1000000 -> wrong args
  CHECK_ACL(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, dim, dim, dim,
                          (const void *)alpha, (const void *)xDevice, -1,
                          ACL_FLOAT16, (const void *)yDevice, -1, ACL_FLOAT16,
                          (const void *)beta, (void *)zDevice, -1, ACL_FLOAT,
                          ACL_COMPUTE_HIGH_PRECISION, stream));

  CHECK_ACL(aclrtSynchronizeStream(stream));
  auto end_acl = std::chrono::high_resolution_clock::now();
  double duration_acl =
      ((double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_acl -
                                                                    start_acl)
           .count()) /
      1e12;

  std::cout << "end aclblas kernel" << std::endl;
  std::cout << "Time taken by aclblas: " << duration_acl << " seconds"
            << std::endl;

  CHECK_ACL(aclrtMemcpy(zHost2, outputByteSize, zDevice, outputByteSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  float maxAbsDiff = 0.0f;
  float maxRelDiff = 0.0f;
  maxDifference((float *)zHost, (float *)zHost2, dim * dim, maxAbsDiff,
                maxRelDiff);
  std::cout << "Between impl. matmul and aclblas matmul \nmax abs difference is: "
            << maxAbsDiff << "\n"
            << "max rel difference is: " << maxRelDiff << "%" << std::endl;
  maxDifference((float *)cRefHost, (float *)zHost, dim * dim, maxAbsDiff,
                maxRelDiff);
  std::cout << "Between impl. matmul and numpy matmul \nmax abs difference is: "
            << maxAbsDiff << "\n"
            << "max rel difference is: " << maxRelDiff << "%" << std::endl;
  maxDifference((aclFloat16 *)cHalfRefHost, (float *)zHost, dim * dim, maxAbsDiff,
                maxRelDiff);
  std::cout << "Between impl. matmul and numpy matmul fp 16 \nmax abs difference is: "
            << maxAbsDiff << "\n"
            << "max rel difference is: " << maxRelDiff << "%" << std::endl;
  maxDifference((float *)cRefHost, (float *)zHost2, dim * dim, maxAbsDiff,
                maxRelDiff);
  std::cout << "Between aclblas matmul and numpy matmul \nmax abs difference is: "
            << maxAbsDiff << "\n"
            << "max rel difference is: " << maxRelDiff << "%" << std::endl;
  maxDifference((aclFloat16 *)cHalfRefHost, (float *)zHost2, dim * dim, maxAbsDiff,
                maxRelDiff);
  std::cout << "Between aclblas matmul and numpy fp16 matmul \nmax abs difference is: "
            << maxAbsDiff << "\n"
            << "max rel difference is: " << maxRelDiff << "%" << std::endl;

  writeNumbersToTextFile(C_path, (float *)zHost, dim, dim);
  writeNumbersToTextFile(C_aclblas_path, (float *)zHost2, dim, dim);

  CHECK_ACL(aclrtFree(xDevice));
  CHECK_ACL(aclrtFree(yDevice));
  CHECK_ACL(aclrtFree(zDevice));
  CHECK_ACL(aclrtFreeHost(xHost));
  CHECK_ACL(aclrtFreeHost(yHost));
  CHECK_ACL(aclrtFreeHost(zHost));
  CHECK_ACL(aclrtFreeHost(tiling));
  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());

  delete[] cRefHost;
  delete[] cHalfRefHost;
#else
#endif
  return 0;
}
