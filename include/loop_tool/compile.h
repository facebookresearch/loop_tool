/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "ir.h"

#include <functional>
#include <unordered_map>
#include <vector>

#define MAX_DEPTH 16

namespace loop_tool {

struct Allocation {
  size_t size;
  int idx;
  bool should_init;
  float init_val;        // TODO don't hardcode float type
  LoopTree::TreeRef lca; // can easily deduce required vars this way
  LoopTree::TreeRef producer;
};

// unfortunately, there is some required auxiliary information
struct Auxiliary {
  std::unordered_map<IR::VarRef, int> var_idx; // index into "tails" array
  std::unordered_map<LoopTree::TreeRef, size_t>
      inner_size; // total size of inner loops over same var
  std::unordered_map<IR::NodeRef, Allocation>
      allocs; // intermediate allocations
  std::unordered_map<LoopTree::TreeRef, std::vector<Allocation>>
      resets; // allocation LCAs
};

// recursively generate functions for loops/nodes(leaves) of the loop tree
using InnerFnType = std::function<void(const std::vector<void *> &,
                                       int[MAX_DEPTH], int[MAX_DEPTH])>;

std::vector<std::pair<int, size_t>> gen_idx_vector(const LoopTree &lt,
                                                   const Allocation &alloc,
                                                   LoopTree::TreeRef use);
std::function<size_t(int[MAX_DEPTH])> gen_idx_func(const LoopTree &lt,
                                                   const Allocation &alloc,
                                                   LoopTree::TreeRef use);
void gen_alloc(const LoopTree &lt, Auxiliary &aux, LoopTree::TreeRef ref);
void exec(const LoopTree &lt, const std::vector<void *> &memory);

Auxiliary calculate_aux(const LoopTree &lt);
std::pair<std::function<void(const std::vector<void *> &)>, std::vector<size_t>>
compile(const LoopTree &lt);
bool trivially_parallel(const LoopTree &lt, LoopTree::TreeRef ref);

} // namespace loop_tool
