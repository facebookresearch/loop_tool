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

using InnerFnType =
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
    inline int64_t size(int start_idx = 0) const {
      int64_t s = 1;
      for (int i = start_idx; i < sizes.size(); ++i) {
        const auto &s_ = sizes.at(i);
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
    // alloc (base vars) mapped to expr, max
    std::vector<symbolic::Expr> full_exprs;
    std::vector<symbolic::Expr> scoped_exprs;
    std::vector<std::pair<int64_t, int64_t>> bounds;

    // DEPRECATED
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

  // optionally always true, this is for cleanup
  mutable bool set_called = false;
  size_t count;
  LoopTree lt;
  std::unordered_map<LoopTree::TreeRef, int64_t>
      inner_sizes;  // total size of inner loops over same var
  std::unordered_map<IR::NodeRef, Allocation> allocations;
  std::unordered_map<IR::NodeRef, IR::NodeRef> resolved_reads;
  std::unordered_map<IR::NodeRef, IR::NodeRef> resolved_writes;
  std::unordered_map<IR::VarRef, int64_t> var_sizes;
  std::unordered_map<IR::VarRef, symbolic::Symbol> var_to_sym;
  std::unordered_map<symbolic::Symbol, IR::VarRef,
                     symbolic::Hash<symbolic::Symbol>>
      sym_to_var;

  Compiler(const LoopTree &lt_);

  Allocation gen_alloc(IR::NodeRef node_ref) const;

  std::pair<std::vector<symbolic::Expr>, std::vector<symbolic::Expr>>
  gen_index_equations(IR::NodeRef read_node_ref, IR::NodeRef write_node_ref,
                      LoopTree::TreeRef ref) const;
  // given a node used at point "ref", generate access information
  Access gen_access(IR::NodeRef node, LoopTree::TreeRef ref) const;

  // DEPRECATED, see gen_index_equations
  std::vector<symbolic::Constraint> gen_constraints(
      IR::NodeRef node, LoopTree::TreeRef ref) const;

  InnerFnType gen_reset(LoopTree::TreeRef ref) const;

  symbolic::Expr reify_sizes(const symbolic::Expr &expr) const;
  int64_t get_expr_max(const symbolic::Expr &) const;
  int64_t get_expr_min(const symbolic::Expr &) const;
  IdxInformation gen_idx_info(LoopTree::TreeRef ref,
                              const Compiler::Access &access) const;
  std::function<int64_t(int indices[MAX_DEPTH])> gen_idx_fn(
      LoopTree::TreeRef ref, const Access &access) const;

  InnerFnType gen_mem_node(LoopTree::TreeRef ref) const;
  InnerFnType gen_add_node(LoopTree::TreeRef ref) const;
  InnerFnType gen_mul_node(LoopTree::TreeRef ref) const;
  InnerFnType gen_binary_node(LoopTree::TreeRef ref) const;
  InnerFnType gen_unary_node(LoopTree::TreeRef ref) const;
  InnerFnType gen_node(LoopTree::TreeRef ref) const;

  InnerFnType gen_loop(
      LoopTree::TreeRef ref,
      std::unordered_map<IR::VarRef, int> overrides) const;

  InnerFnType gen_exec(
      LoopTree::TreeRef ref = -1,
      std::unordered_map<IR::VarRef, int> overrides = {}) const;

  bool is_input_output(IR::NodeRef nr) const;
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

struct CPUBackend : public Backend {
  CPUBackend() : Backend("cpu") {}
  ~CPUBackend() {}
  CPUBackend(std::string name)
      : Backend(name) {}

  std::unique_ptr<Compiled> compile_impl(
      const LoopTree &lt, const std::unordered_set<LoopTree::TreeRef> &parallel,
      LoopTree::TreeRef root) override;
  int hardware_requirement() const override;
};

}  // namespace loop_tool
