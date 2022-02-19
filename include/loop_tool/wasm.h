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
  std::unordered_set<IR::NodeRef> local_storage;
  mutable std::unordered_map<IR::NodeRef, int> local_f32;
  mutable std::unordered_map<IR::NodeRef, int> local_v128;
  mutable std::unordered_map<LoopTree::TreeRef, int> iterators;
  mutable std::unordered_map<IR::NodeRef, int32_t> memory_locations;

 public:
  WebAssemblyCompiler(const LoopTree& lt)
      : Compiler(lt), cg(std::make_shared<wasmblr::CodeGenerator>()) {
    // Logic to calculate "local storage" opportunities.  This effectively means
    // "in-register" and requires static information about things like vector
    // lanes and which register data lives in
    // TODO this is overly restrictive -- we can also store things in register
    // if only their relevant vars are unrolled (and irrelevant are not).
    auto completely_unrolled = [&](LoopTree::TreeRef ref,
                                   LoopTree::TreeRef lca) {
      ref = lt.parent(ref);
      while (ref != lca) {
        if (lt.annotation(ref) != "unroll") {
          return false;
        }
        ref = lt.parent(ref);
      }
      return true;
    };
    for (const auto& node_ref : lt.ir.nodes()) {
      // forced to use real memory in these cases
      if (!lt.scheduled.count(node_ref) ||
          lt.ir.node(node_ref).op() == Operation::write) {
        continue;
      }
      auto alloc = allocations.at(node_ref);
      bool scheduled_consumers = true;
      bool unrolled = true;
      for (const auto& consumer_ref : lt.ir.node(node_ref).outputs()) {
        if (!lt.scheduled.count(consumer_ref)) {
          scheduled_consumers = false;
          break;
        }
        if (!completely_unrolled(lt.scheduled.at(consumer_ref), alloc.lca)) {
          unrolled = false;
          break;
        }
      }
      // we cannot address this memory statically (will need runtime
      // information)
      if (!scheduled_consumers || !unrolled) {
        continue;
      }
      local_storage.insert(node_ref);
    }
  }

  void push_access_to_stack(IR::NodeRef node_ref, LoopTree::TreeRef ref) const;
  void push_float_to_stack(IR::NodeRef node_ref, LoopTree::TreeRef ref) const;
  void push_vector_to_stack(IR::NodeRef node_ref, LoopTree::TreeRef ref) const;
  void emit_vectorized_node(LoopTree::TreeRef ref) const;
  void emit_node(LoopTree::TreeRef ref) const;
  void emit_reset(LoopTree::TreeRef ref) const;
  void emit_loop(LoopTree::TreeRef ref,
                 std::unordered_map<IR::VarRef, int> overrides) const;
  void emit(LoopTree::TreeRef ref,
            std::unordered_map<IR::VarRef, int> overrides) const;
  std::vector<uint8_t> emit() const;
};

}  // namespace loop_tool
