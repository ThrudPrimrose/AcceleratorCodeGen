#pragma once

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
#endif

void writeMatrixToBinaryFile(const void *matrix, const std::string &fileName,
                             const size_t M, const size_t N,
                             const size_t floatSize) {
  std::ofstream outFile(fileName, std::ios::binary);
  if (!outFile) {
    std::cerr << "Could not open the file for writing!" << std::endl;
    return;
  }

  const char *data = reinterpret_cast<const char *>(matrix);
  // Write the matrix data to the file
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      outFile.write(data + m * N * floatSize, N * floatSize);
    }
  }

  outFile.close();

  if (!outFile.good()) {
    std::cerr << "Error occurred while writing to the file!" << std::endl;
  }
}

void readMatrixFromBinaryFile(void *matrix, const std::string &fileName,
                              const size_t M, const size_t N,
                              const size_t floatSize) {
  std::ifstream inFile(fileName, std::ios::binary);
  if (!inFile) {
    std::cerr << "Could not open the file for reading!" << std::endl;
    return;
  }

  char *data = reinterpret_cast<char *>(matrix);

  for (size_t m = 0; m < M; ++m) {
    inFile.read(data + m * N * floatSize, N * floatSize);
  }

  inFile.close();

  if (!inFile.good()) {
    std::cerr << "Error occurred while reading from the file!" << std::endl;
    return;
  }
}

void maxDifference(const float *array1, const float *array2, size_t n,
                   float &maxAbsDiff, float &maxRelDiff) {
  float max_diff = 0.0f;
  float max_rel_diff = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    float diff = std::fabs(array1[i] - array2[i]);
    float rel_diff = 100 * diff / std::fabs((array1[i] + array2[i]) / 2.0);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (rel_diff > max_rel_diff) {
      max_rel_diff = rel_diff;
    }
  }

  maxAbsDiff = max_diff;
  maxRelDiff = max_rel_diff;
}

size_t hash_matrix(const float *matrix, const int M, const int N) {
  size_t hashValue = 0;
  size_t prime = 31;
  bool allZero = true;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float elem = matrix[i * N + j];
      if (elem != 0) {
        allZero = false;
      }
      hashValue = hashValue * prime + std::hash<float>()(elem);
    }
  }

  return allZero ? 0 : hashValue;
}

void writeNumbersToTextFile(const std::string &filename, const float *numbers,
                            int numRows, int numCols) {
  std::ofstream file(filename);
  file << std::fixed;

  float f = 0.f;
  if (file.is_open()) {
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
        f = static_cast<float>(numbers[i * numCols + j]);
        file << std::setprecision(2) << f;
        if (j != numCols - 1) {
          file << " ";
        }
      }
      file << "\n";
    }

    file.close();
  } else {
    std::cerr << "Error: Unable to open file " << filename << " for writing."
              << std::endl;
  }
}

#ifndef __CCE_KT_TEST__

void maxDifference(const aclFloat16 *array1, const aclFloat16 *array2, size_t n,
                   float &maxAbsDiff, float &maxRelDiff) {
  float max_diff = 0.0f;
  float max_rel_diff = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    float f1 = aclFloat16ToFloat(array1[i]);
    float f2 = aclFloat16ToFloat(array2[i]);
    float diff = std::fabs(f1 - f2);
    float rel_diff = 100 * diff / std::fabs((f1 + f2) / 2.0);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (rel_diff > max_rel_diff) {
      max_rel_diff = rel_diff;
    }
  }

  maxAbsDiff = max_diff;
  maxRelDiff = max_rel_diff;
}

void writeNumbersToTextFile(const std::string &filename,
                            const aclFloat16 *numbers, int numRows,
                            int numCols) {
  std::ofstream file(filename);
  file << std::fixed;

  float f = 0.f;
  if (file.is_open()) {
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
        f = aclFloat16ToFloat(numbers[i * numCols + j]);
        file << std::setprecision(2) << f;
        if (j != numCols - 1) {
          file << " ";
        }
      }
      file << "\n";
    }

    file.close();
  } else {
    std::cerr << "Error: Unable to open file " << filename << " for writing."
              << std::endl;
  }
}

size_t hash_matrix(const aclFloat16 *matrix, const int M, const int N) {
  size_t hashValue = 0;
  size_t prime = 31;
  bool allZero = true;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float elem = aclFloat16ToFloat(matrix[i * N + j]);
      if (elem != 0) {
        allZero = false;
      }
      hashValue = hashValue * prime + std::hash<float>()(elem);
    }
  }

  return allZero ? 0 : hashValue;
}
#endif