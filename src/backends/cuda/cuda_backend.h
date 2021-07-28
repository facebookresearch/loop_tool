/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <unordered_map>
#include <unordered_set>

#include "loop_tool/compile.h"
#include "loop_tool/error.h"
#include "loop_tool/hardware.h"
#include "loop_tool/ir.h"

namespace loop_tool {

struct CudaAux {
  // maps loops to the inner size of other threaded loops
  std::unordered_map<LoopTree::TreeRef, int> threaded;
  std::unordered_set<LoopTree::TreeRef> unrolled;
  int threads_per_warp;
  int threads_per_block;
  std::unordered_map<IR::NodeRef, size_t> alloc_threads;
  std::unordered_map<LoopTree::TreeRef, int> syncs;
  std::unordered_map<IR::VarRef, int> tail;  // temporary
};

CudaAux calc_cuda_aux(const LoopTree &lt, const Auxiliary &aux,
                      const std::unordered_set<LoopTree::TreeRef> &threaded);

}  // namespace loop_tool

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    ASSERT(0) << cudaGetErrorString(code) << " " << file << ":" << line;
  }
}

#define CUDA_SAFE_CALL(x)                                       \
  do {                                                          \
    CUresult result = x;                                        \
    const char *msg;                                            \
    cuGetErrorName(result, &msg);                               \
    ASSERT(result == CUDA_SUCCESS)                              \
        << "\nerror: " #x " failed with error " << msg << '\n'; \
  } while (0)
