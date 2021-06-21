/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "compile.h"
#include "error.h"
#include "ir.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <unordered_set>

namespace detail {
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);

    // Mainly for demonstration purposes, i.e. works but is overly simple
    // In the real world, use sth. like boost.hash_combine
    return h1 ^ h2;
  }
};
} // namespace detail
// for manual unrolling/vectorization
using UnrollMap =
    std::unordered_map<std::pair<IR::VarRef, int>, int, detail::pair_hash>;
using UnrollSet =
    std::unordered_set<std::pair<IR::VarRef, int>, detail::pair_hash>;

struct CudaAux {
  // maps loops to the inner size of other threaded loops
  std::unordered_map<LoopTree::TreeRef, int> threaded;
  std::unordered_set<LoopTree::TreeRef> unrolled;
  int threads_per_warp;
  int threads_per_block;
  std::unordered_map<IR::NodeRef, size_t> alloc_threads;
  std::unordered_map<LoopTree::TreeRef, int> syncs;
  std::unordered_map<IR::VarRef, int> tail; // temporary
};

struct CompiledCuda {
  char *ptx;
  CUfunction kernel;
  std::string code;
  size_t num_blocks = 0;
  size_t num_threads = 0;

  size_t peak_bandwidth_gb = 0;

  CUcontext context;
  CUmodule module;
  CUdevice cuDevice;

  CompiledCuda(const LoopTree &lt,
               const std::unordered_set<LoopTree::TreeRef> &threaded);
  ~CompiledCuda();
  // if sync is true, call cuCtxSynchronize
  void operator()(const std::vector<void *> &memory, bool sync = true) const;
};

CudaAux calc_cuda_aux(const LoopTree &lt, const Auxiliary &aux,
                      const std::unordered_set<LoopTree::TreeRef> &threaded);

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    ASSERT(0);
  }
}
