/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "ir.h"

#define MAX_DEPTH 16

namespace loop_tool {

struct Allocation {
  size_t size;
  size_t thread_size;
  int idx;
  bool should_init;
  float init_val;         // TODO don't hardcode float type
  LoopTree::TreeRef lca;  // can easily deduce required vars this way
  LoopTree::TreeRef producer;
};

// unfortunately, there is some required auxiliary information
struct Auxiliary {
  std::unordered_map<IR::VarRef, int> var_idx;  // index into "tails" array
  std::unordered_map<LoopTree::TreeRef, size_t>
      inner_size;  // total size of inner loops over same var
  std::unordered_map<IR::NodeRef, Allocation>
      allocs;  // intermediate allocations
  // thread -> [(idx, stride), ...]
  std::unordered_map<LoopTree::TreeRef, std::vector<std::pair<int, int>>>
      thread_memory;
  std::unordered_map<LoopTree::TreeRef, std::vector<Allocation>>
      resets;  // allocation LCAs
};

// recursively generate functions for loops/nodes(leaves) of the loop tree
using InnerFnType = std::function<void(const std::vector<void *> &,
                                       int[MAX_DEPTH], int[MAX_DEPTH])>;
using GenFnType = std::function<InnerFnType(const LoopTree &, const Auxiliary &,
                                            LoopTree::TreeRef)>;

// returns pairs loop depth, inner size for the var at that depth
//   assuming indices is a map from the loop depth to the current loop
//   iteration, index = indices[p.first] * p.second for p in idx_vec
std::vector<std::pair<int, size_t>> gen_idx_vector(const LoopTree &lt,
                                                   const Auxiliary &aux,
                                                   const Allocation &alloc,
                                                   LoopTree::TreeRef use);
std::function<size_t(int[MAX_DEPTH])> gen_idx_func(const LoopTree &lt,
                                                   const Auxiliary &aux,
                                                   const Allocation &alloc,
                                                   LoopTree::TreeRef use);
void gen_alloc(const LoopTree &lt, Auxiliary &aux, LoopTree::TreeRef ref);
void exec(const LoopTree &lt, const std::vector<void *> &memory);

Auxiliary calculate_aux(const LoopTree &lt);
std::pair<std::function<void(const std::vector<void *> &)>, std::vector<size_t>>
compile(const LoopTree &lt,
        std::function<InnerFnType(const LoopTree &, const Auxiliary &,
                                  LoopTree::TreeRef)>
            callback = {});
bool trivially_parallel(const LoopTree &lt, LoopTree::TreeRef ref);

struct CPUBackend : public Backend {
  CPUBackend() : Backend("cpu") {}
  CPUBackend(std::string name, GenFnType callback_)
      : Backend(name), callback(callback_) {}

  // for easy CPU overwriting
  // TODO map annotations to GenFnTypes
  GenFnType callback = {};

  std::unique_ptr<Compiled> compile_impl(
      const LoopTree &lt, const std::unordered_set<LoopTree::TreeRef> &parallel,
      LoopTree::TreeRef root) override;
  int hardware_requirement() const override;
};

}  // namespace loop_tool
