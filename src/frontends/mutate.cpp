/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <loop_tool/mutate.h>
#include <loop_tool/measure.hpp>
#include <loop_tool/backend.h>

#include <algorithm>
#include <string>

namespace loop_tool {

std::vector<std::string> get_available_actions(const LoopTree& lt, LoopTree::TreeRef ref){
  std::vector<std::string> avail_actions;

  LoopTree::TreeRef p_ref = previous_ref(lt, ref);
  LoopTree::TreeRef n_ref = next_ref(lt, ref);

  if (p_ref == ref){
    p_ref = -1;
  }
  if (n_ref == ref){
    n_ref = -1;
  }

  // General operations
  if (p_ref != -1){
    avail_actions.push_back("up");
  }
  if (n_ref != -1){
    avail_actions.push_back("down");
  }

  // Loop operations
  if (lt.kind(ref) == LoopTree::LOOP){
    // split
    int max_vec_size = 1;

    // the most inner-loop
    if (all_of(lt.children(ref).begin(), lt.children(ref).end(), 
          [&] (int i) {return lt.kind(i) == LoopTree::NODE;})
        ){
      max_vec_size = isa_traits<DABUN_ISA>().vector_size;
    }
    auto loop_iter = lt.loop(ref).size;
    while (loop_iter >= max_vec_size){
      avail_actions.push_back("split_" + std::to_string(loop_iter)); 
      loop_iter /= 2; 
    }

    // merge
    if(p_ref != -1 && lt.kind(p_ref) == LoopTree::LOOP && lt.loop(ref).var == lt.loop(p_ref).var){
      avail_actions.push_back("merge");
    }

    // swap loops
    if (p_ref != -1 && lt.kind(p_ref) == LoopTree::LOOP){
      avail_actions.push_back("swap_up");
    } 
    if (n_ref != -1 && lt.kind(n_ref) == LoopTree::LOOP){
      avail_actions.push_back("swap_down");
    } 

    // Loop annotations
    avail_actions.push_back("vectorize");
    avail_actions.push_back("unroll");
  }
  if (lt.kind(ref) == LoopTree::NODE){
    
    // swap nodes
    if (p_ref != -1 && lt.kind(p_ref) == LoopTree::NODE){
      avail_actions.push_back("swap_up");
    } 
    if (n_ref != -1 && lt.kind(n_ref) == LoopTree::NODE){
      avail_actions.push_back("swap_down");
    }

    // copy input
    for (int i=0; i < get_inputs(lt, ref).size(); i++){
      avail_actions.push_back("copy_input_" + std::to_string(i));
    } 

    // increase - decrease reuse
    avail_actions.push_back("increase_reuse");

    avail_actions.push_back("decrease_reuse");
  }



  return avail_actions;
}


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
  new_ir.delete_node(node_ref);
  return new_ir;
}

// new IR is generated
IR split_var(const IR& ir_, IR::VarRef v) {
  ASSERT(0 && "not yet implemented");
  auto ir = ir_;
  return ir;
}

IR swap_vars(const IR& ir_, IR::NodeRef node_ref, IR::VarRef a, IR::VarRef b) {
  auto ir = ir_;
  auto& node = ir.node(node_ref);
  ASSERT(a != b) << "cannot swap var with itself";
  auto& vars = node.vars();
  auto a_idx = std::find(vars.begin(), vars.end(), a);
  ASSERT(a_idx != vars.end()) << "cannot find var " << ir.var(a).name()
                              << " in node " << ir.dump(node_ref);
  auto b_idx = std::find(vars.begin(), vars.end(), b);
  ASSERT(b_idx != vars.end()) << "cannot find var " << ir.var(b).name()
                              << " in node " << ir.dump(node_ref);
  std::iter_swap(a_idx, b_idx);
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

LoopTree subtree(const LoopTree& lt, LoopTree::TreeRef ref,
                 std::unordered_map<IR::NodeRef, IR::NodeRef> node_map,
                 std::unordered_map<IR::VarRef, IR::VarRef> var_map) {
  auto keep_nodes = [&]() {
    if (lt.kind(ref) == LoopTree::NODE) {
      return std::unordered_set<IR::NodeRef>(lt.node(ref));
    }
    return to_set(collect_nodes(lt, ref));
  }();
  IR new_ir;
  auto nodes = lt.ir.nodes();
  for (auto nr : nodes) {
    const auto& n = lt.ir.node(nr);
    // absorb unscheduled inputs
    if (!lt.scheduled.count(nr)) {
      const auto& outputs = n.outputs();
      bool used_by_keep_nodes = true;
      for (const auto& o : outputs) {
        if (!keep_nodes.count(o)) {
          used_by_keep_nodes = false;
        }
      }
      if (used_by_keep_nodes) {
        keep_nodes.insert(nr);
      }
    }
    if (!keep_nodes.count(nr)) {
      continue;
    }
    std::vector<IR::NodeRef> inputs;
    std::vector<IR::VarRef> vars;
    std::vector<symbolic::Constraint> constraints = n.constraints();
    std::unordered_map<int, IR::VarRef> sym_var_map;

    for (auto inp : n.inputs()) {
      if (keep_nodes.count(inp)) {
        inputs.emplace_back(node_map.at(inp));
      }
    }
    for (auto v : n.vars()) {
      if (!var_map.count(v)) {
        var_map[v] = new_ir.create_var(lt.ir.var(v).name());
      }
      vars.emplace_back(var_map.at(v));
    }
    if (inputs.size() == 0 && n.op() != Operation::read) {
      const auto inp = new_ir.create_node(Operation::read, {}, vars);
      inputs.emplace_back(inp);
      new_ir.add_input(inp);
    }
    for (const auto& p : n.sym_to_var()) {
      sym_var_map[p.first] = var_map.at(p.second);
    }
    node_map[nr] =
        new_ir.create_node(n.op(), inputs, vars, constraints, sym_var_map);

    if (n.op() == Operation::read) {
      new_ir.add_input(node_map[nr]);
    }
    bool write = true;
    if (n.op() == Operation::write) {
      new_ir.add_output(node_map[nr]);
      write = false;
    }
    for (auto out : n.outputs()) {
      if (keep_nodes.count(out)) {
        write = false;
      }
    }
    if (write) {
      const auto out =
          new_ir.create_node(Operation::write, {node_map.at(nr)}, vars);
      new_ir.add_output(out);
    }
  }
  ASSERT(new_ir.outputs().size() > 0);
  ASSERT(new_ir.inputs().size() > 0);
  for (auto nr : nodes) {
    if (!keep_nodes.count(nr)) {
      continue;
    }
    std::vector<std::pair<IR::VarRef, IR::LoopSize>> order;
    std::vector<std::string> annotations;
    const auto& old_order = lt.ir.order(nr);
    const auto& old_annot = lt.ir.loop_annotations(nr);
    for (auto i = 0; i < old_order.size(); ++i) {
      const auto& p = old_order.at(i);
      // NB: ignore removed scheduled loops
      if (!var_map.count(p.first)) {
        std::cerr << "WARNING, REUSE DISABLE NOT PRESERVED\n";
        continue;
      }
      order.emplace_back(var_map.at(p.first), p.second);
      annotations.emplace_back(old_annot.at(i));
    }
    auto new_nr = node_map.at(nr);
    new_ir.set_order(new_nr, order, annotations);
    new_ir.annotate(new_nr, lt.ir.annotation(nr));
    // TODO: this should skip the skipped indices above
    for (auto i : lt.ir.not_reusable(nr)) {
      new_ir.disable_reuse(new_nr, i);
    }
  }

  return LoopTree(new_ir);
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

std::vector<IR::NodeRef> get_inputs(const LoopTree& lt, LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  auto node_ref = lt.node(ref);
  auto& node = lt.ir.node(node_ref);
  return node.inputs();
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

LoopTree swap_vars(const LoopTree& lt, IR::NodeRef node_ref, IR::VarRef a,
                   IR::VarRef b) {
  return LoopTree(swap_vars(lt.ir, node_ref, a, b));
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
  auto annot = lt.annotation(add);
  ASSERT(lt.ir.reduction_vars(node_ref).size() == 0)
      << "cannot add orthogonal inner loop to reduction";
  auto new_ir = lt.ir;
  auto order = new_ir.order(node_ref);
  order.emplace_back(loop.var, IR::LoopSize{loop.size, loop.tail});
  new_ir.set_order(node_ref, order);
  auto new_lt = LoopTree(new_ir);
  auto new_ref = map_ref(new_lt, ref, lt);
  return annotate(lt, new_ref, annot);
}

LoopTree swap_loops(const LoopTree& lt, LoopTree::TreeRef a,
                    LoopTree::TreeRef b) {
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

LoopTree try_swap(const LoopTree& lt, LoopTree::TreeRef a,
                  LoopTree::TreeRef b) {
  bool a_is_loop = lt.kind(a) == LoopTree::LOOP;
  bool b_is_loop = lt.kind(b) == LoopTree::LOOP;
  if (!a_is_loop && !b_is_loop) {
    return swap_nodes(lt, a, b);
  } else if (a_is_loop && b_is_loop) {
    if (lt.loop(a).var != lt.loop(b).var) {
      return swap_loops(lt, a, b);
    }
  } else if (!a_is_loop && b_is_loop) {
    // we're hoisting a node
    if (lt.parent(a) == b) {
      return remove_loop(lt, a, b);
    } else {
      return add_loop(lt, a, b);
    }
  }
  return lt;
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

double eval_runtime(const LoopTree &lt){
  auto c = Compiler(lt);
  auto sizes = c.memory_sizes(true);
  std::vector<void *> memory;
  std::vector<std::vector<float>> data;

  for (int i = 0; i < lt.ir.inputs().size() + lt.ir.outputs().size(); i++){
    data.emplace_back(std::vector<float>(sizes[i]));
  }
  
  for (const auto &v: data){
    memory.emplace_back((void *)(v.data()));
  }

  
  auto cc = getDefaultBackend()->compile(lt);

  // TODO: Run 100 times and get mean, std:
  unsigned iterations = 100;
  unsigned warmup_iterations = 5;
  return dabun::measure_median(
    [&]() {cc->run(memory);}, iterations, warmup_iterations
    );
}


int64_t FLOPs(const LoopTree& lt) {
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

double FLOPS(const LoopTree& lt) {
  return FLOPs(lt) / eval_runtime(lt);
}


bool is_trivially_parallel(const LoopTree& lt, LoopTree::TreeRef ref) {
  bool trivially_parallel = true;
  if (lt.kind(ref) == LoopTree::NODE) {
    return false;
  }
  auto tree_v = lt.loop(ref).var;
  lt.walk(
      [&](LoopTree::TreeRef ref, int) {
        if (lt.kind(ref) == LoopTree::LOOP) {
          return;
        }
        auto node_ref = lt.node(ref);
        bool iters_over = false;
        for (auto v : lt.ir.loop_vars(node_ref)) {
          if (v == tree_v) {
            iters_over = true;
            break;
          }
        }
        if (iters_over) {
          bool pointwise = false;
          for (auto v : lt.ir.pointwise_vars(node_ref)) {
            if (v == tree_v) {
              pointwise = true;
            }
          }
          if (!pointwise) {
            trivially_parallel = false;
          }
        }
      },
      ref);
  return trivially_parallel;
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

LoopTree annotate(
    const LoopTree& lt,
    std::unordered_map<LoopTree::TreeRef, std::string> annotations) {
  auto new_ir = lt.ir;
  for (const auto& p : annotations) {
    auto ref = p.first;
    auto annot = p.second;
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

LoopTree maximize_reuse(const LoopTree& lt) {
  auto ir = lt.ir;

  // This is a naive weighting mechanism
  // we're going to put all reduction vars inward
  std::unordered_map<IR::VarRef, float> weight;
  for (const auto& node_ref : ir.nodes()) {
    const auto& vars = ir.node(node_ref).vars();
    auto pw_vars = to_set(ir.pointwise_vars(node_ref));
    auto reduction_vars = to_set(ir.reduction_vars(node_ref));
    for (auto i = 0; i < vars.size(); ++i) {
      const auto& v = vars.at(i);
      if (pw_vars.count(v)) {
        weight[v] += 1;
      } else if (reduction_vars.count(v)) {
        weight[v] -= 1;
      }
    }
  }

  for (const auto& node_ref : ir.nodes()) {
    auto order = ir.order(node_ref);
    std::stable_sort(order.begin(), order.end(),
                     [&](const std::pair<IR::VarRef, IR::LoopSize>& a,
                         const std::pair<IR::VarRef, IR::LoopSize>& b) {
                       return weight[a.first] > weight[b.first];
                     });
    ir.set_order(node_ref, order);
  }

  return LoopTree(ir);
}

LoopTree unroll_inner_loops(const LoopTree& lt, int32_t unroll_amount) {
  std::unordered_map<LoopTree::TreeRef, std::string> annotations;
  std::vector<LoopTree::TreeRef> no_unroll_above;
  for (const auto& node_ref : lt.ir.nodes()) {
    if (!lt.scheduled.count(node_ref)) {
      continue;
    }
    auto ref = lt.scheduled.at(node_ref);
    auto p = lt.parent(ref);
    int64_t size = (p != -1) ? lt.loop(p).size : 1;
    while (p != -1 && size < unroll_amount) {
      annotations[p] = "unroll";
      p = lt.parent(p);
      size *= (p != -1) ? lt.loop(p).size : 1;
    }
    no_unroll_above.emplace_back(p);
  }
  for (auto p : no_unroll_above) {
    while (p != -1) {
      if (annotations.count(p)) {
        annotations.erase(p);
      }
      p = lt.parent(p);
    }
  }

  return annotate(lt, annotations);
}

std::vector<IR::NodeRef> find(const IR& ir, Operation op) {
  std::vector<IR::NodeRef> out;
  for (const auto& n : ir.nodes()) {
    const auto& node = ir.node(n);
    if (node.op() == op) {
      out.emplace_back(n);
    }
  }
  return out;
}

}  // namespace loop_tool
