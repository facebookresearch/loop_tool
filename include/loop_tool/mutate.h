/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once
#include <loop_tool/loop_tool.h>

namespace loop_tool {

IR split_node(const IR& ir, IR::NodeRef node_ref,
              std::vector<IR::VarRef> injected);
IR split_var(const IR& ir, IR::VarRef v);

LoopTree split(const LoopTree& lt, LoopTree::TreeRef ref, int64_t size);
// merges upward
LoopTree merge(const LoopTree& lt, LoopTree::TreeRef ref);
LoopTree copy_input(const LoopTree& lt, LoopTree::TreeRef ref, int idx);
LoopTree delete_copy(const LoopTree& lt, LoopTree::TreeRef ref);
LoopTree swap(const LoopTree& lt, LoopTree::TreeRef a, LoopTree::TreeRef b);
LoopTree add_loop(const LoopTree& lt, LoopTree::TreeRef ref,
                  LoopTree::TreeRef add);
LoopTree remove_loop(const LoopTree& lt, LoopTree::TreeRef ref,
                     LoopTree::TreeRef rem);
LoopTree swap_nodes(const LoopTree& lt, LoopTree::TreeRef a,
                    LoopTree::TreeRef b);
LoopTree disable_reuse(const LoopTree& lt, LoopTree::TreeRef loop,
                       IR::NodeRef n);
LoopTree enable_reuse(const LoopTree& lt, LoopTree::TreeRef loop,
                      IR::NodeRef n);
LoopTree decrease_reuse(const LoopTree& lt, LoopTree::TreeRef ref);
LoopTree increase_reuse(const LoopTree& lt, LoopTree::TreeRef ref);
LoopTree::TreeRef next_ref(const LoopTree& lt, LoopTree::TreeRef ref);
LoopTree::TreeRef previous_ref(const LoopTree& lt, LoopTree::TreeRef ref);
int64_t flops(const LoopTree& lt);
LoopTree annotate(const LoopTree& lt, LoopTree::TreeRef ref, std::string annot);
// map an old ref to a close new ref after mutation, return the new ref
LoopTree::TreeRef map_ref(const LoopTree& new_lt, LoopTree::TreeRef old_ref,
                          const LoopTree& old_lt);

LoopTree maximize_reuse(const LoopTree& lt);
LoopTree unroll_inner_loops(const LoopTree& lt, int32_t unroll_amount);

}  // namespace loop_tool
