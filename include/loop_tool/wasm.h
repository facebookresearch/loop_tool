/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "loop_tool/compile.h"
#include "wasmblr.h"

namespace loop_tool {

class WebAssemblyCompiler : public Compiler {
  mutable std::shared_ptr<wasmblr::CodeGenerator> cg;
  std::unordered_set<IR::NodeRef> stack_storage;
  std::unordered_set<IR::NodeRef> local_storage;
  std::unordered_map<IR::NodeRef, IR::VarRef> stack_vector_storage;
  std::unordered_map<IR::NodeRef, IR::VarRef> local_vector_storage;
  std::unordered_set<LoopTree::TreeRef> vectorized_loops;
  mutable std::unordered_set<IR::NodeRef> stack_f32;
  mutable std::unordered_set<IR::NodeRef> stack_v128;
  mutable int32_t tmp_f32;
  mutable int32_t tmp_v128;
  mutable std::unordered_map<IR::NodeRef, std::vector<int>> local_f32;
  mutable std::unordered_map<IR::NodeRef, std::vector<int>> local_v128;
  mutable std::unordered_map<LoopTree::TreeRef, int> iterators;
  mutable std::unordered_map<IR::NodeRef, int32_t> memory_locations;

 public:
  WebAssemblyCompiler(const LoopTree& lt);

  int64_t get_unroll_offset(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      const std::unordered_map<LoopTree::TreeRef, int32_t>& unrolls) const;
  int64_t get_unroll_offset(
      IR::NodeRef node_ref, LoopTree::TreeRef ref, LoopTree::TreeRef root,
      const symbolic::Expr& idx_expr,
      const std::unordered_map<LoopTree::TreeRef, int32_t>& unrolls) const;

  int32_t push_access_to_stack(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  void push_float_to_stack(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls,
      bool force_memory_load = false) const;
  void push_vector_to_stack(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls, IR::VarRef dim,
      bool force_memory_load = false) const;
  void store_float_from_stack(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  void store_vector_from_stack(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls,
      IR::VarRef dim) const;
  int32_t get_tmp_f32() const;
  int32_t get_tmp_v128() const;

 private:
  bool should_store_stack(IR::NodeRef node_ref) const;
  IR::VarRef should_store_vectorized_dim(IR::NodeRef node_ref) const;

 public:
  bool needs_reset(IR::NodeRef node_ref) const;

  void emit_node(LoopTree::TreeRef ref,
                 std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  void emit_reset(LoopTree::TreeRef ref) const;
  void emit_loop(LoopTree::TreeRef ref,
                 std::unordered_map<IR::VarRef, int> overrides,
                 std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  void emit(LoopTree::TreeRef ref,
            std::unordered_map<IR::VarRef, int> overrides,
            std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  std::vector<uint8_t> emit() const;
  inline bool is_local(IR::NodeRef node_ref) const {
    return local_storage.count(node_ref) ||
           local_vector_storage.count(node_ref);
  }
  inline bool is_on_stack(IR::NodeRef node_ref) const {
    return stack_storage.count(node_ref) ||
           stack_vector_storage.count(node_ref);
  }
  inline bool is_vector_stored(IR::NodeRef node_ref) const {
    return local_vector_storage.count(node_ref) ||
           stack_vector_storage.count(node_ref);
  }
  IR::VarRef vector_storage_dim(IR::NodeRef node_ref) const {
    if (local_vector_storage.count(node_ref)) {
      return local_vector_storage.at(node_ref);
    }
    if (stack_vector_storage.count(node_ref)) {
      return stack_vector_storage.at(node_ref);
    }
    return -1;
  }
  inline bool is_broadcast(IR::NodeRef node_ref) const {
    const auto& vs = lt.ir.node(node_ref).vars();
    auto var = vector_storage_dim(node_ref);
    if (var == -1) {
      return false;
    }
    return !(vs.size() && vs.back() == var);
  }
  bool should_vectorize(LoopTree::TreeRef ref) const;

  void emit_vectorized_node(
      LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  void emit_vectorized_loop(
      LoopTree::TreeRef ref, std::unordered_map<IR::VarRef, int> overrides,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
};

}  // namespace loop_tool
