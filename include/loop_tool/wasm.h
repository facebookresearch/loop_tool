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
  std::unordered_set<IR::NodeRef> local_vector_storage;
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

  std::unordered_map<symbolic::Symbol,
                     std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                     symbolic::Hash<symbolic::Symbol>>
  get_symbol_strides(
      LoopTree::TreeRef ref, LoopTree::TreeRef root,
      const std::unordered_map<LoopTree::TreeRef, int32_t>& unrolls) const;

  symbolic::Expr get_scoped_expr(const Compiler::Access& access) const;

  int32_t push_access_to_stack(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  void push_float_to_stack(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  void push_vector_to_stack(
      IR::NodeRef node_ref, LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
  int32_t get_tmp_f32() const;
  bool needs_reset(IR::NodeRef node_ref) const;
  void emit_vectorized_node(
      LoopTree::TreeRef ref,
      std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const;
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
  bool is_local(IR::NodeRef node_ref) const {
    return local_storage.count(node_ref);
  }
};

}  // namespace loop_tool
