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
  virtual ~Compiler() = default;

  Allocation gen_alloc(IR::NodeRef node_ref) const;

  std::pair<std::vector<symbolic::Expr>, std::vector<symbolic::Expr>>
  gen_index_equations(IR::NodeRef read_node_ref, IR::NodeRef write_node_ref,
                      LoopTree::TreeRef ref) const;
  // given a node used at point "ref", generate access information
  Access gen_access(IR::NodeRef node, LoopTree::TreeRef ref) const;

  symbolic::Expr get_scoped_expr(const Access &access) const;
  std::unordered_map<symbolic::Symbol,
                     std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                     symbolic::Hash<symbolic::Symbol>>
  get_symbol_strides(LoopTree::TreeRef ref, LoopTree::TreeRef root) const;

  std::function<float *(const std::vector<void *> &memory,
                        int indices[MAX_DEPTH])>
  gen_access_fn(const Access &access, LoopTree::TreeRef ref) const;
  std::vector<std::pair<symbolic::Expr, int64_t>> get_constraints(
      const Access &access) const;

  InnerFnType gen_reset(LoopTree::TreeRef ref) const;

  symbolic::Expr reify_sizes(const symbolic::Expr &expr) const;
  int64_t get_expr_max(const symbolic::Expr &) const;
  int64_t get_expr_min(const symbolic::Expr &) const;

  InnerFnType gen_mem_node(LoopTree::TreeRef ref) const;
  InnerFnType gen_binary_node(LoopTree::TreeRef ref) const;
  InnerFnType gen_unary_node(LoopTree::TreeRef ref) const;
  InnerFnType gen_node(LoopTree::TreeRef ref) const;

  InnerFnType gen_loop(LoopTree::TreeRef ref,
                       std::unordered_map<IR::VarRef, int> overrides) const;

  InnerFnType gen_exec(
      LoopTree::TreeRef ref = -1,
      std::unordered_map<IR::VarRef, int> overrides = {}) const;
  virtual std::string gen_string() const;

  std::vector<void *> allocate() const;
  std::vector<int64_t> memory_sizes() const;
};

struct CPUInterpretedBackend : public Backend {
  CPUInterpretedBackend() : Backend("cpu_interpreted") {}
  ~CPUInterpretedBackend() {}
  CPUInterpretedBackend(std::string name) : Backend(name) {}

  std::unique_ptr<Compiled> compile_impl(const LoopTree &lt) override;
  int hardware_requirement() const override;
};

}  // namespace loop_tool
