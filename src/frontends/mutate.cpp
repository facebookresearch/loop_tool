/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <loop_tool/mutate.h>

#include <algorithm>

namespace loop_tool {

IR split_node(const IR& ir, IR::NodeRef node_ref,
              std::vector<IR::VarRef> injected) {
  IR new_ir = ir;
  auto& node = new_ir.node(node_ref);
  auto vs_vec = new_ir.loop_vars(node_ref);
  std::unordered_set<IR::VarRef> vs{vs_vec.begin(), vs_vec.end()};
  for (auto v : injected) {
    ASSERT(vs.count(v));
    vs.erase(v);
  }
  ASSERT(vs.size() > 0);
  auto new_node_ref = new_ir.create_node(node.op(), {}, node.vars());
  new_ir.replace_all_uses(node_ref, new_node_ref);
  new_ir.update_vars(node_ref, injected);
  new_ir.update_inputs(new_node_ref, {node_ref});
  new_ir.reset_aux(node_ref);
  new_ir.reset_aux(new_node_ref);
  return new_ir;
}

IR duplicate_input(const IR& ir, IR::NodeRef node_ref, int idx) {
  IR new_ir = ir;
  auto& node = new_ir.node(node_ref);
  ASSERT(node.inputs().size() > idx) << "cannot get input at index " << idx;
  auto inp_ref = node.inputs().at(idx);
  auto& inp_node = new_ir.node(inp_ref);
  auto new_node_ref = new_ir.create_node(Operation::copy, {}, inp_node.vars());
  new_ir.set_order(new_node_ref, new_ir.order(node_ref),
                   new_ir.loop_annotations(node_ref));
  new_ir.replace_all_uses(inp_ref, new_node_ref);
  new_ir.update_inputs(new_node_ref, {inp_ref});
  return new_ir;
}

IR remove_copy(const IR& ir, IR::NodeRef node_ref) {
  IR new_ir = ir;
  auto& node = new_ir.node(node_ref);
  ASSERT(node.op() == Operation::copy);
  auto inp_ref = node.inputs().at(0);
  new_ir.replace_all_uses(node_ref, inp_ref);
  return new_ir;
}

// new IR is generated
IR split_var(const IR& ir_, IR::VarRef v) {
  ASSERT(0 && "not yet implemented");
  auto ir = ir_;
  return ir;
}

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

int64_t get_inner_size(const LoopTree& lt, LoopTree::TreeRef ref) {
  auto loop = lt.loop(ref);
  int64_t inner_size = 0;
  lt.walk(
      [&](LoopTree::TreeRef iref, int) {
        if (lt.kind(iref) == LoopTree::LOOP) {
          return;
        }
        int64_t cur = 1;
        auto p = lt.parent(iref);
        while (p != ref) {
          const auto& l = lt.loop(p);
          if (l.var == loop.var) {
            cur *= l.size;
            cur += l.tail;
          }
          p = lt.parent(p);
        }
        inner_size = std::max(cur, inner_size);
      },
      ref);
  return inner_size;
}

LoopTree split(const LoopTree& lt, LoopTree::TreeRef ref, int64_t size) {
  ASSERT(lt.kind(ref) == LoopTree::LOOP) << "can only split loops";
  auto node_refs = collect_nodes(lt, ref);
  auto loop = lt.loop(ref);

  ASSERT(loop.size / size > 0)
      << "attempting to split a loop of size " << loop.size << " by " << size;
  // collect all inner loops
  auto inner_size = get_inner_size(lt, ref);
  auto new_size = (loop.size * inner_size + loop.tail) / (size * inner_size);
  auto new_tail = (loop.size * inner_size + loop.tail) % (size * inner_size);
  ASSERT(new_size > 0) << "invalid split size";

  auto split0 = std::make_pair(loop.var, IR::LoopSize{new_size, new_tail});
  auto split1 = std::make_pair(loop.var, IR::LoopSize{size, 0});

  auto replace = [&](IR::NodeRef node_ref, int idx) {
    std::vector<std::pair<IR::VarRef, IR::LoopSize>> new_order;
    std::vector<std::string> new_annot;
    auto old_order = lt.ir.order(node_ref);
    auto old_annot = lt.ir.loop_annotations(node_ref);
    for (auto i = 0; i < old_order.size(); ++i) {
      if (i == idx) {
        new_order.emplace_back(split0);
        new_annot.emplace_back();
        new_order.emplace_back(split1);
        new_annot.emplace_back();
        continue;
      }
      new_order.emplace_back(old_order.at(i));
      new_annot.emplace_back(old_annot.at(i));
    }
    return std::make_pair(new_order, new_annot);
  };

  auto new_ir = lt.ir;
  for (auto node_ref : node_refs) {
    auto loop_order = lt.loop_order(node_ref);
    for (auto i = 0; i < loop_order.size(); ++i) {
      if (loop_order.at(i) == loop) {
        const auto& r = replace(node_ref, i);
        new_ir.set_order(node_ref, r.first, r.second);
      }
    }
  }
  return LoopTree(new_ir);
}

LoopTree merge(const LoopTree& lt, LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::LOOP);
  auto loop = lt.loop(ref);
  auto p = lt.parent(ref);
  auto parent_loop = (p != -1) ? lt.loop(p) : lt.loop(ref);
  while (p != -1 && parent_loop.var != loop.var) {
    p = lt.parent(p);
    if (p != -1) {
      parent_loop = lt.loop(p);
    }
  }
  if (p == -1) {
    return lt;
  }

  auto inner_size = get_inner_size(lt, ref);
  auto total_size = loop.size * inner_size + loop.tail;
  total_size = parent_loop.size * total_size + parent_loop.tail;
  auto new_size = total_size / inner_size;
  auto new_tail = total_size % inner_size;
  auto merged = std::make_pair(loop.var, IR::LoopSize{new_size, new_tail});

  auto replace = [&](IR::NodeRef node_ref, int merge_idx, int del_idx) {
    std::vector<std::pair<IR::VarRef, IR::LoopSize>> new_order;
    std::vector<std::string> new_annot;
    auto old_order = lt.ir.order(node_ref);
    auto old_annot = lt.ir.loop_annotations(node_ref);
    for (auto i = 0; i < old_order.size(); ++i) {
      if (i == merge_idx) {
        new_order.emplace_back(merged);
        new_annot.emplace_back();
        continue;
      } else if (i == del_idx) {
        continue;
      }
      new_order.emplace_back(old_order.at(i));
      new_annot.emplace_back(old_annot.at(i));
    }
    return std::make_pair(new_order, new_annot);
  };

  auto node_refs = collect_nodes(lt, p);
  auto new_ir = lt.ir;
  for (const auto& node_ref : node_refs) {
    auto loop_order = lt.loop_order(node_ref);
    int merge_idx = -1;
    int del_idx = -1;
    for (auto i = 0; i < loop_order.size(); ++i) {
      auto l = loop_order.at(i);
      if (l == parent_loop) {
        merge_idx = i;
      } else if (l == loop) {
        del_idx = i;
      }
    }
    if (merge_idx > -1 && del_idx > -1) {
      const auto& r = replace(node_ref, merge_idx, del_idx);
      new_ir.set_order(node_ref, r.first, r.second);
    }
  }
  return LoopTree(new_ir);
}

LoopTree copy_input(const LoopTree& lt, LoopTree::TreeRef ref, int idx) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  auto node_ref = lt.node(ref);
  ASSERT(idx >= 0) << "cannot use negatively indexed input";
  auto new_ir = duplicate_input(lt.ir, node_ref, idx);
  return LoopTree(new_ir);
}

LoopTree delete_copy(const LoopTree& lt, LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  auto node_ref = lt.node(ref);
  auto new_ir = remove_copy(lt.ir, node_ref);
  return LoopTree(new_ir);
}

// update the priority of nodes so that they're topologically swapped
LoopTree swap_nodes(const LoopTree& lt, LoopTree::TreeRef a,
                    LoopTree::TreeRef b) {
  ASSERT(lt.kind(a) == LoopTree::NODE);
  ASSERT(lt.kind(b) == LoopTree::NODE);
  auto a_nr = lt.node(a);
  auto b_nr = lt.node(b);
  auto new_ir = lt.ir;
  auto p_a = new_ir.priority(a_nr);
  auto p_b = new_ir.priority(b_nr);
  if (p_a == p_b) {
    p_a = p_b + 0.1;
  }
  new_ir.set_priority(a_nr, p_b);
  new_ir.set_priority(b_nr, p_a);
  return LoopTree(new_ir);
}

// this isn't always safe in the case of view operations
LoopTree remove_loop(const LoopTree& lt, LoopTree::TreeRef ref,
                     LoopTree::TreeRef rem) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  ASSERT(lt.kind(rem) == LoopTree::LOOP);
  auto node_ref = lt.node(ref);
  auto loop = lt.loop(rem);
  const auto& node = lt.ir.node(node_ref);
  if (node.op() != Operation::view) {
    auto needed_vars = to_set(lt.ir.loop_vars(node_ref));
    ASSERT(!needed_vars.count(loop.var))
        << "attempting to deschedule a necessary loop";
  }
  auto loop_order = lt.loop_order(node_ref);
  auto idx = -1;
  for (auto i = 0; i < loop_order.size(); ++i) {
    if (loop_order.at(i) == loop) {
      idx = i;
    }
  }
  auto new_ir = lt.ir;
  if (idx >= 0) {
    auto order = new_ir.order(node_ref);
    order.erase(order.begin() + idx);
    new_ir.set_order(node_ref, order);
  }
  return LoopTree(new_ir);
}

LoopTree add_loop(const LoopTree& lt, LoopTree::TreeRef ref,
                  LoopTree::TreeRef add) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  ASSERT(lt.kind(add) == LoopTree::LOOP);
  auto node_ref = lt.node(ref);
  auto loop = lt.loop(add);
  ASSERT(lt.ir.reduction_vars(node_ref).size() == 0)
      << "cannot add orthogonal inner loop to reduction";
  auto new_ir = lt.ir;
  auto order = new_ir.order(node_ref);
  order.emplace_back(loop.var, IR::LoopSize{loop.size, loop.tail});
  new_ir.set_order(node_ref, order);
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
    // can't swap tail params
    ASSERT(0) << "swapping the same var is not supported, resplit instead";
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
    if (a_idx < 0 || b_idx < 0) {
      continue;
    }
    auto order = lt.ir.order(node_ref);
    auto loop_annotations = lt.ir.loop_annotations(node_ref);
    std::iter_swap(order.begin() + a_idx, order.begin() + b_idx);
    std::iter_swap(loop_annotations.begin() + a_idx,
                   loop_annotations.begin() + b_idx);
    new_ir.set_order(node_ref, order, loop_annotations);
  }
  return LoopTree(new_ir);
}

LoopTree toggle_reuse(const LoopTree& lt, LoopTree::TreeRef ref, IR::NodeRef n,
                      bool enable) {
  auto new_ir = lt.ir;
  auto loop_order = lt.loop_order(n);
  for (auto i = 0; i < loop_order.size(); ++i) {
    const auto& loop = loop_order.at(i);
    if (loop == lt.loop(ref)) {
      if (enable) {
        new_ir.enable_reuse(n, i);
      } else {
        new_ir.disable_reuse(n, i);
      }
      break;
    }
  }
  return LoopTree(new_ir);
}

LoopTree disable_reuse(const LoopTree& lt, LoopTree::TreeRef loop,
                       IR::NodeRef n) {
  return toggle_reuse(lt, loop, n, false);
}

LoopTree enable_reuse(const LoopTree& lt, LoopTree::TreeRef loop,
                      IR::NodeRef n) {
  return toggle_reuse(lt, loop, n, true);
}

LoopTree decrease_reuse(const LoopTree& lt, LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  auto node_ref = lt.node(ref);
  auto new_ir = lt.ir;
  const auto& not_reusable = new_ir.not_reusable(node_ref);
  const auto& order = new_ir.order(node_ref);
  if (not_reusable.size()) {
    for (auto i = 0; i < order.size() - 1; ++i) {
      if (not_reusable.count(i + 1)) {
        new_ir.disable_reuse(node_ref, i);
        break;
      }
    }
  } else {
    new_ir.disable_reuse(node_ref, order.size() - 1);
  }
  return LoopTree(new_ir);
}

LoopTree increase_reuse(const LoopTree& lt, LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  auto node_ref = lt.node(ref);
  auto new_ir = lt.ir;
  const auto& not_reusable = new_ir.not_reusable(node_ref);
  const auto& order = new_ir.order(node_ref);
  for (auto i = 0; i < order.size(); ++i) {
    if (not_reusable.count(i)) {
      new_ir.enable_reuse(node_ref, i);
      break;
    }
  }
  return LoopTree(new_ir);
}

LoopTree::TreeRef next_ref_impl(const LoopTree& lt, LoopTree::TreeRef ref,
                                bool handle_children) {
  if (ref == -1) {
    return -1;
  }
  auto children = lt.children(ref);
  if (children.size() && handle_children) {
    return children.at(0);
  }
  auto siblings = lt.children(lt.parent(ref));
  auto idx = 0;
  while (siblings[idx] != ref) {
    idx++;
  }
  idx++;
  if (idx < siblings.size()) {
    return siblings.at(idx);
  }
  auto next = next_ref_impl(lt, lt.parent(ref), false);
  if (next == -1) {
    return -1;
  }
  return next;
}

LoopTree::TreeRef next_ref(const LoopTree& lt, LoopTree::TreeRef ref) {
  auto next = next_ref_impl(lt, ref, true);
  return (next == -1) ? ref : next;
}

LoopTree::TreeRef previous_ref(const LoopTree& lt, LoopTree::TreeRef ref) {
  if (ref == -1) {
    return ref;
  }
  auto siblings = lt.children(lt.parent(ref));
  auto idx = 0;
  while (siblings[idx] != ref) {
    idx++;
  }
  idx--;
  if (idx < 0) {
    auto parent = lt.parent(ref);
    if (parent == -1) {
      return ref;
    }
    return parent;
  }
  auto next = siblings.at(idx);
  auto trailing_next = next;
  while (next != ref) {
    trailing_next = next;
    next = next_ref(lt, next);
  }
  return trailing_next;
}

int64_t flops(const LoopTree& lt) {
  int64_t total = 0;
  std::vector<LoopTree::Loop> running_loops;
  auto fn = [&](const LoopTree::TreeRef& ref, int depth) {
    running_loops.resize(depth);
    if (lt.kind(ref) == LoopTree::LOOP) {
      running_loops.emplace_back(lt.loop(ref));
      return;
    }
    std::unordered_map<IR::VarRef, int64_t> var_sizes;
    for (auto iter = running_loops.rbegin(); iter != running_loops.rend();
         ++iter) {
      if (var_sizes.count(iter->var) == 0) {
        var_sizes[iter->var] = 1;
      }
      var_sizes[iter->var] *= iter->size;
      var_sizes[iter->var] += iter->tail;
    }
    int64_t iters = 1;
    for (const auto& p : var_sizes) {
      iters *= p.second;
    }
    switch (lt.ir.node(lt.node(ref)).op()) {
      case Operation::copy:
      case Operation::read:
      case Operation::write:
      case Operation::view:
        return;
      default:
        break;
    }
    total += iters;
  };
  lt.walk(fn);
  return total;
}

LoopTree annotate(const LoopTree& lt, LoopTree::TreeRef ref,
                  std::string annot) {
  auto new_ir = lt.ir;
  if (lt.kind(ref) == LoopTree::NODE) {
    auto node_ref = lt.node(ref);
    new_ir.annotate(node_ref, annot);
    return LoopTree(new_ir);
  }
  auto loop = lt.loop(ref);
  auto node_refs = collect_nodes(lt, ref);
  for (const auto& node_ref : node_refs) {
    auto loop_order = lt.loop_order(node_ref);
    for (auto i = 0; i < loop_order.size(); ++i) {
      if (loop_order.at(i) == loop) {
        new_ir.annotate_loop(node_ref, i, annot);
      }
    }
  }
  return LoopTree(new_ir);
}

LoopTree::TreeRef map_ref(const LoopTree& new_lt, LoopTree::TreeRef old_ref,
                          const LoopTree& old_lt) {
  if (old_lt.kind(old_ref) == LoopTree::NODE) {
    auto node_ref = old_lt.node(old_ref);
    if (new_lt.scheduled.count(node_ref)) {
      return new_lt.scheduled.at(node_ref);
    }
    // unscheduled nodes
    return new_lt.roots.at(0);
  }
  auto loop = old_lt.loop(old_ref);
  auto count = 0;
  auto count_shallow = 0;
  auto version = 0;
  auto version_shallow = 0;
  old_lt.walk([&](LoopTree::TreeRef ref, int) {
    if (old_lt.kind(ref) != LoopTree::LOOP) {
      return;
    }
    auto old_loop = old_lt.loop(ref);
    if (ref == old_ref) {
      version = count;
      version_shallow = count_shallow;
    }
    if (old_loop.var == loop.var) {
      if (old_loop.var_depth == loop.var_depth) {
        count++;
      }
      if (old_loop.var_depth == (loop.var_depth + 1)) {
        count_shallow++;
      }
    }
  });
  // we want to figure out which version of var + depth we have

  std::vector<LoopTree::TreeRef> new_versions;
  std::vector<LoopTree::TreeRef> new_versions_shallow;
  new_lt.walk([&](LoopTree::TreeRef ref, int) {
    if (new_lt.kind(ref) != LoopTree::LOOP) {
      return;
    }
    auto new_loop = new_lt.loop(ref);
    if (new_loop.var == loop.var && new_loop.var_depth == loop.var_depth) {
      new_versions.emplace_back(ref);
    }
    if (new_loop.var == loop.var &&
        (new_loop.var_depth + 1) == loop.var_depth) {
      new_versions_shallow.emplace_back(ref);
    }
  });
  if (new_versions.size() == 0) {
    // weird edge case
    if (new_versions_shallow.size() == 0) {
      return new_lt.roots.at(0);
    }
    auto new_idx = std::min(version, (int32_t)new_versions_shallow.size() - 1);
    return new_versions_shallow.at(new_idx);
  }
  auto new_idx = std::min(version, (int32_t)new_versions.size() - 1);
  return new_versions.at(new_idx);
}

}  // namespace loop_tool
