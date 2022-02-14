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
  mutable std::unordered_map<IR::NodeRef, int> locals;
  mutable std::unordered_map<IR::NodeRef, int32_t> memory_locations;
  mutable std::unordered_map<LoopTree::TreeRef, int> iterators;

 public:
  WebAssemblyCompiler(const LoopTree& lt)
      : Compiler(lt), cg(std::make_shared<wasmblr::CodeGenerator>()) {}

  void push_access_to_stack(IR::NodeRef node_ref, LoopTree::TreeRef ref) const;
  void emit_node(LoopTree::TreeRef ref) const;
  void emit_reset(LoopTree::TreeRef ref) const;
  void emit_loop(LoopTree::TreeRef ref,
                 std::unordered_map<IR::VarRef, int> overrides) const;
  void emit(LoopTree::TreeRef ref,
            std::unordered_map<IR::VarRef, int> overrides) const;
  std::vector<uint8_t> emit() const;
};

}  // namespace loop_tool
