/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <loop_tool/mutate.h>

#include <algorithm>

namespace loop_tool {

std::vector<IR::NodeRef> collect_nodes(const LoopTree& lt,
                                       LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::LOOP)
      << "can only collect nodes within loops";
  std::vector<IR::NodeRef> node_refs;
  lt.walk(
      [&](LoopTree::TreeRef tr, int depth) {
        if (lt.kind(tr) == LoopTree::NODE) {
          node_refs.emplace_back(lt.node(tr));
        }
      },
      ref);
  return node_refs;
}

LoopTree split(const LoopTree& lt, LoopTree::TreeRef ref, int64_t size) {
  ASSERT(lt.kind(ref) == LoopTree::LOOP) << "can only split loops";
  auto node_refs = collect_nodes(lt, ref);
  auto loop = lt.loop(ref);

  ASSERT(loop.size / size > 0);
  auto new_size = loop.size / size;
  auto new_tail = loop.size % size;
  auto split0 = std::make_pair(loop.var, IR::LoopSize{new_size, new_tail});
  auto split1 = std::make_pair(loop.var, IR::LoopSize{size, 0});

  auto replace = [&](IR::NodeRef node_ref, int idx) {
    std::vector<std::pair<IR::VarRef, IR::LoopSize>> new_order;
    auto old_order = lt.ir.order(node_ref);
    for (auto i = 0; i < old_order.size(); ++i) {
      if (i == idx) {
        new_order.emplace_back(split0);
        new_order.emplace_back(split1);
        continue;
      }
      new_order.emplace_back(old_order.at(i));
    }
    return new_order;
  };

  auto new_ir = lt.ir;
  for (auto node_ref : node_refs) {
    auto loop_order = lt.loop_order(node_ref);
    for (auto i = 0; i < loop_order.size(); ++i) {
      if (loop_order.at(i) == loop) {
        new_ir.set_order(node_ref, replace(node_ref, i));
      }
    }
  }
  return LoopTree(new_ir);
}

LoopTree swap(const LoopTree& lt, LoopTree::TreeRef a, LoopTree::TreeRef b) {
  ASSERT(lt.kind(a) == LoopTree::LOOP);
  ASSERT(lt.kind(b) == LoopTree::LOOP);
  bool a_is_parent = false;
  bool b_is_parent = false;
  lt.walk(
      [&](LoopTree::TreeRef ref, int) {
        if (ref == b) {
          a_is_parent = true;
        }
      },
      a);
  lt.walk(
      [&](LoopTree::TreeRef ref, int) {
        if (ref == a) {
          b_is_parent = true;
        }
      },
      b);
  ASSERT(a_is_parent ^ b_is_parent);
  if (b_is_parent) {
    auto tmp = a;
    a = b;
    b = tmp;
  }

  auto node_refs = collect_nodes(lt, a);
  auto a_loop = lt.loop(a);
  auto b_loop = lt.loop(b);
  if (a_loop.var == b_loop.var) {
    // gotta swap tail params
    ASSERT(0) << "swapping the same var is not yet supported, resplit instead";
  }
  auto new_ir = lt.ir;
  for (auto node_ref : node_refs) {
    auto loop_order = lt.loop_order(node_ref);
    auto a_idx = -1;
    auto b_idx = -1;
    for (auto i = 0; i < loop_order.size(); ++i) {
      if (loop_order.at(i) == a_loop) {
        a_idx = i;
      }
      if (loop_order.at(i) == b_loop) {
        b_idx = i;
      }
    }
    ASSERT(a_idx >= 0 && b_idx >= 0);
    auto order = lt.ir.order(node_ref);
    std::iter_swap(order.begin() + a_idx, order.begin() + b_idx);
    new_ir.set_order(node_ref, order);
  }
  return LoopTree(new_ir);
}

}  // namespace loop_tool
