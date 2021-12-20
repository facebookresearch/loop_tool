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

#define MAX_DEPTH 8

namespace loop_tool {

using InnerFnTypeImproved =
    std::function<void(const std::vector<void *> &, int[MAX_DEPTH])>;

// Generates runnable code (there's also CodeGenerator, which generates text)
class Compiler {
 public:
  struct Allocation {
    Allocation() = default;
    Allocation(int memory_idx, IR::NodeRef node_ref_)
        : mem_idx(memory_idx), node_ref(node_ref_) {}
    Allocation(int memory_idx, IR::NodeRef node_ref_,
               const std::vector<int64_t> &sizes_, LoopTree::TreeRef lca_)
        : mem_idx(memory_idx), sizes(sizes_), node_ref(node_ref_), lca(lca_) {}
    int mem_idx = -1;
    // scoped sizes
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    inline int64_t size() const {
      int64_t s = 1;
      for (const auto &s_ : sizes) {
        s *= std::max(s_, (int64_t)1);
      }
      return s;
    }
    IR::NodeRef node_ref = -1;
    LoopTree::TreeRef lca = -1;
  };

  struct Access {
    Access(const Allocation &a) : alloc(a) {}
    Allocation alloc;
    // stride, offset, max
    std::unordered_map<IR::VarRef, std::tuple<int64_t, int64_t, int64_t>> vars;
    int64_t total_offset;
  };

  struct IdxInformation {
    std::vector<int64_t> strides;
    int64_t offset = 0;
    // optional overrides
    std::vector<int> idxs;
    std::vector<int64_t> maxes;
    std::vector<int64_t> mins;
  };

  size_t count;
  LoopTree lt;
  std::unordered_map<LoopTree::TreeRef, int64_t>
      inner_sizes;  // total size of inner loops over same var
  std::unordered_map<IR::NodeRef, Allocation> allocations;
  std::unordered_map<IR::NodeRef, IR::NodeRef> resolved_views;
  std::unordered_map<IR::VarRef, int64_t> var_sizes;
  std::unordered_map<IR::VarRef, symbolic::Symbol> var_to_sym;
  std::unordered_map<symbolic::Symbol, IR::VarRef,
                     symbolic::Hash<symbolic::Symbol>>
      sym_to_var;

  Compiler(const LoopTree &lt_);

  Allocation gen_alloc(IR::NodeRef node_ref) const;

  // given a node used at point "ref", generate access information
  Access gen_access(IR::NodeRef node, LoopTree::TreeRef ref) const;

  std::vector<symbolic::Constraint> gen_constraints(
      IR::NodeRef node, LoopTree::TreeRef ref) const;

  InnerFnTypeImproved gen_reset(LoopTree::TreeRef ref) const;

  IdxInformation gen_idx_info(LoopTree::TreeRef ref,
                              const Compiler::Access &access) const;
  std::function<int64_t(int indices[MAX_DEPTH])> gen_idx_fn(
      LoopTree::TreeRef ref, const Access &access) const;

  InnerFnTypeImproved gen_mem_node(LoopTree::TreeRef ref) const;
  InnerFnTypeImproved gen_add_node(LoopTree::TreeRef ref) const;
  InnerFnTypeImproved gen_mul_node(LoopTree::TreeRef ref) const;
  InnerFnTypeImproved gen_binary_node(LoopTree::TreeRef ref) const;
  InnerFnTypeImproved gen_unary_node(LoopTree::TreeRef ref) const;
  InnerFnTypeImproved gen_node(LoopTree::TreeRef ref) const;

  InnerFnTypeImproved gen_loop(
      LoopTree::TreeRef ref,
      std::unordered_map<IR::VarRef, int> overrides) const;

  InnerFnTypeImproved gen_exec(
      LoopTree::TreeRef ref = -1,
      std::unordered_map<IR::VarRef, int> overrides = {}) const;

  std::string gen_access_string(IR::NodeRef node_ref,
                                LoopTree::TreeRef ref) const;
  std::string gen_reset_string(LoopTree::TreeRef ref) const;
  std::string gen_mem_node_string(LoopTree::TreeRef ref) const;
  std::string gen_compute_node_string(LoopTree::TreeRef ref) const;
  inline std::string gen_indent(LoopTree::TreeRef ref, int extra = 0) const {
    auto depth = ((ref == -1) ? 0 : lt.depth(ref) + 1);
    return std::string((depth + extra) * 2, ' ');
  }
  std::string gen_node_string(LoopTree::TreeRef ref) const;
  std::string gen_loop_string(
      LoopTree::TreeRef ref,
      std::unordered_map<IR::VarRef, int> overrides) const;
  std::string gen_string(
      LoopTree::TreeRef ref = -1,
      std::unordered_map<IR::VarRef, int> overrides = {}) const;

  std::vector<void *> allocate() const;
  std::vector<int64_t> memory_sizes() const;
};

struct Allocation {
  int64_t size;
  int64_t thread_size;
  int idx;
  bool should_init;
  float init_val;         // TODO don't hardcode float type
  LoopTree::TreeRef lca;  // can easily deduce required vars this way
  LoopTree::TreeRef producer;
};

// unfortunately, there is some required auxiliary information
struct Auxiliary {
  std::unordered_map<IR::VarRef, int> var_idx;  // index into "tails" array
  std::unordered_map<LoopTree::TreeRef, int64_t>
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
std::vector<std::pair<int, int64_t>> gen_idx_vector(const LoopTree &lt,
                                                    const Auxiliary &aux,
                                                    const Allocation &alloc,
                                                    LoopTree::TreeRef use);
std::function<int64_t(int[MAX_DEPTH])> gen_idx_func(const LoopTree &lt,
                                                    const Auxiliary &aux,
                                                    const Allocation &alloc,
                                                    LoopTree::TreeRef use);
void gen_alloc(const LoopTree &lt, Auxiliary &aux, LoopTree::TreeRef ref);
void exec(const LoopTree &lt, const std::vector<void *> &memory);

Auxiliary calculate_aux(const LoopTree &lt);
std::pair<std::function<void(const std::vector<void *> &)>,
          std::vector<int64_t>>
compile(const LoopTree &lt,
        std::function<InnerFnType(const LoopTree &, const Auxiliary &,
                                  LoopTree::TreeRef)>
            callback = {});
bool trivially_parallel(const LoopTree &lt, LoopTree::TreeRef ref);

struct CPUBackend : public Backend {
  CPUBackend() : Backend("cpu") {}
  ~CPUBackend() {}
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
