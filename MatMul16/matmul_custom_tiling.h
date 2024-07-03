#pragma once

#include <cstdint>

struct MatMulCustomTilingData {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t numPEInXDim;
  uint32_t numPEInYDim;
};