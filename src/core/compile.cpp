/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/compile.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_set>

#include "loop_tool/backend.h"
#include "loop_tool/error.h"
#include "loop_tool/mutate.h"
#include "loop_tool/symbolic.h"

namespace loop_tool {
using namespace symbolic;

using IdxFn = std::function<int64_t(int indices[MAX_DEPTH])>;

size_t getCount() {
  static size_t count = 0;
  return count++;
}

Compiler::Compiler(const LoopTree &lt_) : lt(lt_) {
  count = getCount();

  std::unordered_map<IR::VarRef, int64_t> cur_sizes;
  std::unordered_set<LoopTree::TreeRef> traversed;

  // if a view node is unscheduled, it can still be written to or read from
  // if as_write, resolve as if the view is being written to
  // otherwise read from
  auto resolve_view = [&](IR::NodeRef n, bool as_write) {
    const auto &node = lt.ir.node(n);
    if (node.op() != Operation::view) {
      return n;
    }
    while (!lt.scheduled.count(n)) {
      const auto &node = lt.ir.node(n);
      if (node.op() == Operation::write || node.op() == Operation::read) {
        return n;
      }
      ASSERT(node.op() == Operation::view);
      if (as_write) {
        ASSERT(node.outputs().size() == 1);
        n = node.outputs().at(0);
      } else {
        ASSERT(node.inputs().size() == 1);
        n = node.inputs().at(0);
      }
    }
    return n;
  };

  for (const auto &node_ref : lt.ir.nodes()) {
    resolved_reads[node_ref] = resolve_view(node_ref, false);
    resolved_writes[node_ref] = resolve_view(node_ref, true);
    auto node = lt.ir.node(node_ref);
    auto add_sym = [&](symbolic::Symbol sym) {
      if (node.has_sym(sym)) {
        var_to_sym[node.var(sym)] = sym;
        sym_to_var[sym] = node.var(sym);
      }
    };
    for (const auto &c : node.constraints()) {
      for (auto sym : c.first.symbols()) {
        add_sym(sym);
      }
      for (auto sym : c.second.symbols()) {
        add_sym(sym);
      }
      if (c.first.op() == Op::size) {
        auto sym = c.first.args().at(0);
        auto val = c.second;
        if (sym.type() == Expr::Type::symbol &&
            val.type() == Expr::Type::value) {
          var_sizes[sym_to_var.at(sym.symbol())] = val.value();
        }
      }
    }
    // Sizes can also be defined by user specified loop orders
    auto order = lt.ir.order(node_ref);
    std::reverse(order.begin(), order.end());
    std::unordered_map<IR::VarRef, int64_t> tmp_sizes;
    for (const auto &o : order) {
      if (!tmp_sizes.count(o.first)) {
        tmp_sizes[o.first] = 1;
      }
      tmp_sizes[o.first] *= o.second.size;
      tmp_sizes[o.first] += o.second.tail;
    }
    for (const auto &p : tmp_sizes) {
      if (var_sizes.count(p.first)) {
        var_sizes[p.first] =
            std::max(tmp_sizes.at(p.first), var_sizes.at(p.first));
      } else {
        var_sizes[p.first] = tmp_sizes.at(p.first);
      }
    }
  }
  for (const auto &v : lt.ir.vars()) {
    if (var_to_sym.count(v)) {
      continue;
    }
    auto s = Symbol(lt.ir.var(v).name() + "_gen");
    var_to_sym[v] = s;
    sym_to_var[s] = v;
  }

  for (const auto &v : lt.ir.vars()) {
    ASSERT(var_sizes.count(v))
        << "size could not be deduced for var " << lt.ir.var(v).name();
  }

  // loops we've accounted for
  std::unordered_set<LoopTree::TreeRef> seen;
  for (const auto &n : lt.ir.nodes()) {
    if (!lt.scheduled.count(n)) {
      continue;
    }
    auto ref = lt.scheduled.at(n);
    auto p = lt.parent(ref);
    if (seen.count(p)) {
      continue;
    }
    seen.insert(p);
    std::unordered_map<IR::VarRef, int64_t> cur_sizes;
    while (p != -1) {
      auto loop = lt.loop(p);
      auto inner_size = cur_sizes.count(loop.var) ? cur_sizes.at(loop.var) : 1;
      if (inner_sizes.count(p)) {
        inner_size = std::max(inner_size, inner_sizes.at(p));
      }
      inner_sizes[p] = inner_size;
      int64_t var_size = loop.size * inner_size + loop.tail;
      cur_sizes[loop.var] = var_size;
      if (var_sizes.count(loop.var)) {
        var_size = std::max(var_sizes.at(loop.var), var_size);
      }
      var_sizes[loop.var] = var_size;
      p = lt.parent(p);
    }
  }

  // gen_alloc only works after we get var_sizes
  for (auto node_ref : lt.ir.nodes()) {
    allocations[node_ref] = gen_alloc(node_ref);
  }
}

// algo:
// generate a loop with size + tail for this loop
// if there's an override for this ref, use the specified size/tail
// overrides are just parent loops emiting their tails.
InnerFnType Compiler::gen_loop(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  auto loop = lt.loop(ref);
  ASSERT(loop.size > -1);
  ASSERT(loop.tail > -1);
  int size = loop.size;
  int tail = loop.tail;

  // if there's an override, take it
  if (overrides.count(loop.var)) {
    auto override_size = overrides.at(loop.var);
    auto inner_size = inner_sizes.at(ref);
    size = override_size / inner_size;
    tail = override_size % inner_size;
    overrides.erase(loop.var);
  }

  std::vector<InnerFnType> body_children;
  std::vector<InnerFnType> tail_children;
  for (const auto &cref : lt.children(ref)) {
    body_children.emplace_back(gen_exec(cref, overrides));
  }
  if (tail > 0) {
    // find first loop of same var, and override
    overrides[loop.var] = tail;
    for (const auto &cref : lt.children(ref)) {
      tail_children.emplace_back(gen_exec(cref, overrides));
    }
  }

  auto reset = gen_reset(ref);
  auto idx = lt.depth(ref);
  auto tail_fn = [=](const std::vector<void *> &memory,
                     int indices[MAX_DEPTH]) {
    reset(memory, indices);
    indices[idx] = size;
    for (const auto &c : tail_children) {
      c(memory, indices);
    }
  };

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    for (auto i = 0; i < size; ++i) {
      reset(memory, indices);
      indices[idx] = i;
      for (const auto &c : body_children) {
        c(memory, indices);
      }
    }
    tail_fn(memory, indices);
  };
}

InnerFnType Compiler::gen_reset(LoopTree::TreeRef ref) const {
  std::vector<std::tuple<int, int64_t, float>> resets;
  for (const auto &p : allocations) {
    const auto &alloc = p.second;
    if (alloc.lca == ref) {
      const auto &node = lt.ir.node(alloc.node_ref);
      switch (node.op()) {
        case Operation::add:
          resets.emplace_back(alloc.mem_idx, alloc.size(), 0.0);
          break;
        case Operation::subtract:
          resets.emplace_back(alloc.mem_idx, alloc.size(), 0.0);
          break;
        case Operation::multiply:
          resets.emplace_back(alloc.mem_idx, alloc.size(), 1.0);
          break;
        case Operation::divide:
          resets.emplace_back(alloc.mem_idx, alloc.size(), 1.0);
          break;
        case Operation::min:
          resets.emplace_back(alloc.mem_idx, alloc.size(),
                              std::numeric_limits<float>::max());
          break;
        case Operation::max:
          resets.emplace_back(alloc.mem_idx, alloc.size(),
                              -std::numeric_limits<float>::max());
          break;
        // memory ops
        case Operation::read:
        case Operation::write:
        case Operation::view:
        // unary ops
        case Operation::exp:
        case Operation::sqrt:
        case Operation::negate:
        case Operation::reciprocal:
          break;
        default:
          ASSERT(0) << "cannot generate reset for op: "
                    << lt.ir.dump(alloc.node_ref);
      }
      if (node.op() == Operation::add) {
      } else if (node.op() == Operation::multiply) {
      }
    }
  }
  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    for (const auto &reset : resets) {
      for (int64_t i = 0; i < std::get<1>(reset); ++i) {
        reinterpret_cast<float *>(memory[std::get<0>(reset)])[i] =
            std::get<2>(reset);
      }
    }
  };
}

InnerFnType Compiler::gen_backend_exec(
    LoopTree::TreeRef ref, std::unordered_map<IR::VarRef, int> overrides,
    const std::string &backend) const {
  const auto &backends = getBackends();
  ASSERT(backends.count(backend)) << "Can't find backend " << backend;
  std::unordered_map<IR::NodeRef, IR::NodeRef> node_map;
  std::unordered_map<IR::VarRef, IR::VarRef> var_map;

  auto sub_lt = subtree(lt, ref, node_map, var_map);
  std::unordered_map<int, int> memory_map;
  auto sub_tree_idx = 0;
  auto get_induced_strides = [&](std::vector<IR::VarRef> vars,
                                 std::vector<IR::VarRef> new_vars) {
    // collect all inner vars and generate strides
    // e.g.
    //  [ a, b, c ]
    //  [    b    ]
    // should become
    //  [|b| * |c|, 0, 1]
    std::vector<int64_t> strides(vars.size());
    auto nvs = to_set(new_vars);
    auto get_inner_size = [&](int idx) {
      int64_t size = 1;
      for (auto i = idx; i < vars.size(); ++i) {
        size *= var_sizes.at(vars[i]);
      }
      return size;
    };

    for (auto i = 0; i < vars.size(); ++i) {
      const auto &v = vars.at(i);
      if (var_map.count(v) && nvs.count(var_map.at(v))) {
        strides[i] = 0;
        continue;
      }
      strides[i] = get_inner_size(i + 1);
    }
    return strides;
  };

  // generate a mapping from index to stride for the node
  std::unordered_map<int, int64_t> mutations;
  auto update_mutations = [&](IR::NodeRef old_nr, IR::NodeRef new_nr) {
    ASSERT(node_map.at(old_nr) == new_nr);
    auto depth = lt.depth(ref);
    const auto &node = lt.ir.node(old_nr);
    const auto &sub_node = sub_lt.ir.node(new_nr);
    auto strides = get_induced_strides(node.vars(), sub_node.vars());
    auto sym_strides = get_symbol_strides(ref, -1);

    for (auto i = 0; i < strides.size(); ++i) {
      auto v = node.vars().at(i);
      auto stride = strides.at(i);
      auto sym = var_to_sym.at(v);
      for (const auto &p : sym_strides.at(sym)) {
        if (p.first >= depth) {
          continue;
        }
        mutations[p.first] = p.second * stride;
      }
    }
  };

  for (const auto &sn : sub_lt.ir.inputs()) {
    auto idx = 0;
    bool found = false;
    for (const auto &n : lt.ir.inputs()) {
      if (n == node_map[sn]) {
        memory_map[sub_tree_idx] = idx;
        update_mutations(n, sn);
        found = true;
        break;
      }
      idx++;
    }
    ASSERT(found);
    sub_tree_idx++;
  }
  for (const auto &sn : sub_lt.ir.outputs()) {
    auto idx = 0;
    bool found = false;
    for (const auto &n : lt.ir.outputs()) {
      if (n == node_map[sn]) {
        memory_map[sub_tree_idx] = idx;
        update_mutations(n, sn);
        found = true;
        break;
      }
      idx++;
    }
    ASSERT(found);
    sub_tree_idx++;
  }

  auto cc = std::shared_ptr<Compiled>(backends.at(backend)->compile(sub_lt));
  auto mutate_memory = [memory_map, mutations](
                           int i, const std::vector<void *> &memory,
                           int indices[MAX_DEPTH]) {
    float *base_memory = (float *)memory[memory_map.at(i)];
    for (const auto &p : mutations) {
      base_memory += indices[p.first] * p.second;
    }
    return base_memory;
  };
  auto mem_size = sub_lt.ir.inputs().size() + sub_lt.ir.outputs().size();
  return [cc, mutate_memory, mem_size](const std::vector<void *> &memory,
                                       int indices[MAX_DEPTH]) {
    std::vector<void *> mutated_memory(mem_size);
    for (auto i = 0; i < mutated_memory.size(); ++i) {
      mutated_memory[i] = mutate_memory(i, memory, indices);
    }
    cc->run(mutated_memory);
  };
}

InnerFnType Compiler::gen_exec(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  if (ref == -1) {
    std::vector<InnerFnType> roots;
    for (const auto &cref : lt.roots) {
      roots.emplace_back(gen_exec(cref, overrides));
    }
    auto reset = gen_reset(ref);
    return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
      reset(memory, indices);
      for (const auto &fn : roots) {
        fn(memory, indices);
      }
    };
  }
  auto split_string = [](std::string s) {
    std::stringstream ss(s);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    std::vector<std::string> vstrings(begin, end);
    return vstrings;
  };
  auto annots = split_string(lt.annotation(ref));
  for (const auto &annot : annots) {
    if (annot[0] == '[' && annot[annot.size() - 1] == ']') {
      auto backend = annot.substr(1, annot.size() - 2);
      return gen_backend_exec(ref, overrides, backend);
    }
  }

  if (lt.kind(ref) == LoopTree::NODE) {
    return gen_node(ref);
  }
  ASSERT(lt.kind(ref) == LoopTree::LOOP);
  return gen_loop(ref, overrides);
}

std::string Compiler::gen_string() const {
  std::stringstream ss;
  ss << "[interpreted]\n";
  ss << lt.dump();
  return ss.str();
}

Expr Compiler::get_scoped_expr(const Compiler::Access &access) const {
  Expr full_expr(0);
  for (auto i = 0; i < access.scoped_exprs.size(); ++i) {
    auto stride = access.alloc.size(i + 1);
    const auto &expr = access.scoped_exprs.at(i);
    full_expr = full_expr + expr * Expr(stride);
  }
  full_expr = full_expr.simplify();
  return full_expr;
}

std::unordered_map<Symbol, std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                   Hash<Symbol>>
Compiler::get_symbol_strides(LoopTree::TreeRef ref,
                             LoopTree::TreeRef root) const {
  std::unordered_map<Symbol, std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                     Hash<Symbol>>
      sym_strides;
  auto p = lt.parent(ref);
  while (p != root) {
    const auto &l = lt.loop(p);
    auto sym = var_to_sym.at(l.var);
    auto stride = inner_sizes.at(p);
    sym_strides[sym].emplace_back(p, stride);
    p = lt.parent(p);
  }
  return sym_strides;
}

// pairs of index equation and maximum value (always lower bounded by 0)
// second value -1 implies no upperbound constraint
std::vector<std::pair<Expr, int64_t>> Compiler::get_constraints(
    const Compiler::Access &access) const {
  std::vector<std::pair<Expr, int64_t>> constraints;
  for (auto i = 0; i < access.bounds.size(); ++i) {
    const auto &bound = access.bounds.at(i);
    const auto &expr = access.scoped_exprs.at(i);
    // 0 < expr < -1, default condition (unconstrained)
    if (bound.first == 0 && bound.second == -1) {
      continue;
    }
    if (expr.can_evaluate() && expr.evaluate() == 0) {
      continue;
    }
    constraints.emplace_back(expr, bound.second);
  }
  return constraints;
}

std::vector<int64_t> Compiler::memory_sizes(bool include_io) const {
  std::vector<int64_t> memory(allocations.size());
  for (const auto &p : allocations) {
    const auto &alloc = p.second;
    // don't allocate inputs and outputs
    if (!include_io &&
        (alloc.mem_idx < lt.ir.inputs().size() + lt.ir.outputs().size())) {
      memory[alloc.mem_idx] = 0;
      continue;
    }
    size_t size = 1;
    for (auto s : alloc.sizes) {
      size *= s > 0 ? s : 1;
    }
    memory[alloc.mem_idx] = size;
  }
  return memory;
}

// this includes locally threaded and scoped vars (which reduce strides)
Compiler::Allocation Compiler::gen_alloc(IR::NodeRef node_ref) const {
  const auto &inputs = lt.ir.inputs();
  const auto &outputs = lt.ir.outputs();
  int mem_idx = -1;
  for (auto i = 0; i < inputs.size(); ++i) {
    if (inputs.at(i) == node_ref) {
      mem_idx = i;
    }
  }
  for (auto i = 0; i < outputs.size(); ++i) {
    if (outputs.at(i) == node_ref) {
      mem_idx = i + inputs.size();
    }
  }
  // we need to find a new spot to store this
  if (mem_idx == -1) {
    mem_idx = inputs.size() + outputs.size();
    for (const auto &p : allocations) {
      // these allocations already have a spot
      if (p.second.mem_idx >= (inputs.size() + outputs.size())) {
        mem_idx++;
      }
    }
  }

  const auto &node = lt.ir.node(node_ref);
  if (!lt.scheduled.count(node_ref)) {
    std::vector<int64_t> sizes;
    if (node.op() == Operation::write || node.op() == Operation::read) {
      for (auto v : node.vars()) {
        sizes.emplace_back(var_sizes.at(v));
      }
    }
    return Allocation(mem_idx, node_ref, sizes, -1);
  }

  std::function<std::vector<LoopTree::TreeRef>(IR::NodeRef nr, bool io_switch)>
      get_scheduled_deps;
  get_scheduled_deps = [&](IR::NodeRef nr,
                           bool io_switch) -> std::vector<LoopTree::TreeRef> {
    auto &n = lt.ir.node(nr);
    std::vector<LoopTree::TreeRef> dep_refs;
    for (const auto &dep_ref : (io_switch ? n.inputs() : n.outputs())) {
      if (!lt.scheduled.count(dep_ref)) {
        if (lt.ir.node(dep_ref).op() == Operation::write) {
          dep_refs.emplace_back(-1);
          continue;
        }
        for (auto dep : get_scheduled_deps(dep_ref, io_switch)) {
          dep_refs.emplace_back(dep);
        }
      } else {
        dep_refs.emplace_back(lt.scheduled.at(dep_ref));
      }
    }
    return dep_refs;
  };

  auto ref = lt.parent(lt.scheduled.at(node_ref));
  auto lca = ref;
  for (auto tr : get_scheduled_deps(node_ref, false)) {
    lca = lt.lca(lca, tr);
  }
  if (node.op() == Operation::write || node.op() == Operation::read) {
    lca = -1;
  }

  std::unordered_map<IR::VarRef, int64_t> var_sizes;
  while (ref != lca) {
    auto loop = lt.loop(ref);
    ref = lt.parent(ref);
    if (!var_sizes.count(loop.var)) {
      var_sizes[loop.var] = 1;
    }
    var_sizes[loop.var] *= loop.size;
    var_sizes[loop.var] += loop.tail;
  }
  std::vector<int64_t> sizes;
  for (auto v : node.vars()) {
    if (var_sizes.count(v)) {
      ASSERT(var_sizes.at(v) > 0);
      sizes.emplace_back(var_sizes.at(v));
    } else {
      sizes.emplace_back(1);
    }
  }
  return Allocation(mem_idx, node_ref, sizes, lca);
}

/*
 There are two types of accesses in loop_tool, reads or writes.
 For both there is a necessary calculation of strides and offset, which is what
 this function does. The algorithm is as follows:
 1. determine "real" buffer
 2. collect variables
   a. collect scoped variables
   b. collect node variables (input and output)
   c. find intersection
 3. map scoped variables to "real" buffer variables
   a. for each collected variable, find stride into buffer
   b. for all collected = 0, find offset into buffer
   e.g. x = x' + 1
   if "real" is x and collected is x', then offset is 1
   if "real" is x' and collected is x, then offset is -1 (negative indices are
 always skipped)
*/

// returns two index equation into the read and write nodes (of arbitrary
// dimensions) with respect to the variables available at schedule point "ref"
std::pair<std::vector<Expr>, std::vector<Expr>> Compiler::gen_index_equations(
    IR::NodeRef read_node_ref, IR::NodeRef write_node_ref,
    LoopTree::TreeRef ref) const {
  auto get_chain = [&](IR::NodeRef cur_node_ref, IR::NodeRef target_node_ref) {
    auto prev_node_ref = cur_node_ref;
    std::vector<IR::NodeRef> collected;
    while (prev_node_ref != target_node_ref) {
      ASSERT(lt.ir.node(prev_node_ref).op() == Operation::view)
          << "non-view in chain!";
      collected.emplace_back(prev_node_ref);
      prev_node_ref = lt.ir.node(prev_node_ref).inputs().at(0);
    }
    return collected;
  };
  auto rev_chain = get_chain(write_node_ref, read_node_ref);
  auto chain = rev_chain;
  std::reverse(chain.begin(), chain.end());

  auto get_expr = [&](IR::NodeRef node_ref, IR::VarRef input_var,
                      bool toward_input) {
    const auto &sym = var_to_sym.at(input_var);
    const auto &node = lt.ir.node(node_ref);
    const auto &out_vars = to_set(node.vars());
    const auto &in_vars = node.inputs().size()
                              ? to_set(lt.ir.node(node.inputs().at(0)).vars())
                              : std::unordered_set<IR::VarRef>{};
    const auto &constraints = node.constraints();
    for (const auto &c : constraints) {
      if (c.first == Expr(sym)) {
        bool only_output_vars = true;
        bool only_input_vars = true;
        for (const auto &s : c.second.symbols(false)) {
          if (!out_vars.count(sym_to_var.at(s))) {
            only_output_vars = false;
          }
          if (!in_vars.count(sym_to_var.at(s))) {
            only_input_vars = false;
          }
        }
        if (!toward_input && only_output_vars) {
          return c.second;
        } else if (toward_input && only_input_vars) {
          return c.second;
        }
      }
    }
    // default case there is no mapping for this variable
    return Expr(sym);
  };

  auto avail_syms = [&]() {
    std::unordered_set<Symbol, Hash<Symbol>> out;
    auto cur_ref = lt.parent(ref);
    while (cur_ref != -1) {
      auto var = lt.loop(cur_ref).var;
      ASSERT(var_to_sym.count(var))
          << "Cannot find symbolicated variable " << lt.ir.var(var).name();
      out.insert(var_to_sym.at(var));
      cur_ref = lt.parent(cur_ref);
    }
    return out;
  }();

  std::vector<Expr> base_exprs;
  for (auto v : lt.ir.node(read_node_ref).vars()) {
    base_exprs.emplace_back(get_expr(read_node_ref, v, false));
  }

  std::unordered_set<Symbol, Hash<Symbol>> write_syms;
  std::vector<Expr> base_write_exprs;
  for (auto v : lt.ir.node(write_node_ref).vars()) {
    base_write_exprs.emplace_back(get_expr(write_node_ref, v, false));
    write_syms.insert(var_to_sym.at(v));
  }

  auto collect_fw =
      [&](const std::unordered_set<Symbol, Hash<Symbol>> &target_syms,
          std::vector<Expr> base_exprs) {
        for (auto nr : chain) {
          const auto &node = lt.ir.node(nr);
          ASSERT(node.inputs().size() == 1);
          auto vars = lt.ir.node(node.inputs().at(0)).vars();
          for (const auto &v : vars) {
            auto sym = var_to_sym.at(v);
            if (target_syms.count(sym)) {
              continue;
            }
            auto new_expr = get_expr(nr, v, false);
            for (auto &expr : base_exprs) {
              expr = expr.replace(sym, new_expr).simplify();
              expr = expr.replace(Expr::size(sym), var_sizes.at(v)).simplify();
            }
          }
        }
        return base_exprs;
      };

  auto collect_bw =
      [&](const std::unordered_set<Symbol, Hash<Symbol>> &target_syms,
          std::vector<Expr> base_exprs) {
        for (auto nr : rev_chain) {
          const auto &node = lt.ir.node(nr);
          ASSERT(node.inputs().size() == 1);
          auto vars = node.vars();
          for (const auto &v : vars) {
            auto sym = var_to_sym.at(v);
            if (target_syms.count(sym)) {
              continue;
            }
            auto new_expr = get_expr(nr, v, true);
            for (auto &expr : base_exprs) {
              expr = expr.replace(sym, new_expr).simplify();
              expr = expr.replace(Expr::size(sym), var_sizes.at(v)).simplify();
            }
          }
        }
        return base_exprs;
      };

  auto read_exprs = collect_fw(avail_syms, base_exprs);
  auto write_exprs = collect_bw(avail_syms, base_write_exprs);
  return std::make_pair(read_exprs, write_exprs);
}

Expr Compiler::reify_sizes(const Expr &expr) const {
  return expr
      .walk([&](const Expr &e) {
        if (e.op() == Op::size) {
          ASSERT(e.args().size() == 1);
          auto sym = e.args().at(0).symbol();
          return Expr(var_sizes.at(sym_to_var.at(sym)));
        }
        return e;
      })
      .simplify();
}

int64_t Compiler::get_expr_max(const Expr &expr) const {
  auto no_size = reify_sizes(expr);
  auto max = no_size
                 .walk([&](const Expr &e) {
                   if (e.type() == Expr::Type::symbol) {
                     auto sym = e.symbol();
                     return Expr(std::max(var_sizes.at(sym_to_var.at(sym)) - 1,
                                          (int64_t)0));
                   }
                   return e;
                 })
                 .simplify();
  ASSERT(max.type() == Expr::Type::value)
      << "Couldn't derive explicit upper bound for expr " << expr.dump()
      << " (simplified to " << max.dump() << ")";
  return max.value() + 1;
}

int64_t Compiler::get_expr_min(const Expr &expr) const {
  auto no_size = reify_sizes(expr);
  auto min = no_size
                 .walk([&](const Expr &e) {
                   if (e.type() == Expr::Type::symbol) {
                     return Expr(0);
                   }
                   return e;
                 })
                 .simplify();
  ASSERT(min.type() == Expr::Type::value)
      << "Couldn't derive explicit lower bound for expr " << expr.dump()
      << " (simplified to " << min.dump() << ")";
  return min.value();
}

Compiler::Access Compiler::gen_access(IR::NodeRef node_ref,
                                      LoopTree::TreeRef ref) const {
  auto read_node_ref = resolved_reads.at(node_ref);
  auto view_exprs = gen_index_equations(read_node_ref, node_ref, ref);

  bool is_write = node_ref == lt.node(ref);
  for (const auto &e : view_exprs.second) {
    ASSERT(!is_write || e.type() == Expr::Type::symbol)
        << "viewed writes not yet supported, found expr: " << e.dump();
  }

  const auto &read_node = lt.ir.node(read_node_ref);
  auto alloc = allocations.at(read_node_ref);

  auto use_node_ref = lt.node(ref);
  const auto &use_node = lt.ir.node(use_node_ref);

  auto node_vars = to_set(lt.ir.all_vars(use_node_ref));
  auto scope_vars = lt.scope_vars(ref);
  auto vars = intersection(node_vars, scope_vars);

  // either input vars
  std::vector<symbolic::Symbol> read_symbols;
  for (auto v : read_node.vars()) {
    if (var_to_sym.count(v)) {
      read_symbols.emplace_back(var_to_sym.at(v));
    }
  }
  auto read_exprs = view_exprs.first;
  for (auto i = 0; i < read_exprs.size(); ++i) {
    const auto &e = read_exprs.at(i);
  }

  ASSERT(alloc.sizes.size() == read_exprs.size());
  auto stride_at = [&](int idx) {
    int64_t stride = alloc.sizes.at(idx) > 0 ? 1 : 0;
    for (auto i = idx + 1; i < alloc.sizes.size(); ++i) {
      auto size = alloc.sizes.at(i);
      stride *= size > 0 ? size : 1;
    }
    return stride;
  };

  Access access(alloc);

  std::unordered_set<Symbol, Hash<Symbol>> sym_in_scope;
  auto p = lt.parent(ref);
  while (p != alloc.lca) {
    const auto &l = lt.loop(p);
    auto sym = var_to_sym.at(l.var);
    sym_in_scope.insert(sym);
    p = lt.parent(p);
  }

  for (auto i = 0; i < read_exprs.size(); ++i) {
    const auto &expr = reify_sizes(read_exprs.at(i));
    auto min = get_expr_min(expr);
    auto max = get_expr_max(expr);
    if (max <= alloc.sizes.at(i)) {
      max = -1;  // this means we don't need to check anything
    } else {
      max = alloc.sizes.at(i);
      ASSERT(max > 0);
    }
    access.full_exprs.emplace_back(expr);
    access.bounds.emplace_back(min, max);
  }

  for (const auto &full_expr : access.full_exprs) {
    auto expr = full_expr
                    .walk([&](const Expr &e) {
                      if (e.type() == Expr::Type::symbol) {
                        if (!sym_in_scope.count(e.symbol())) {
                          return Expr(0);
                        }
                      }
                      return e;
                    })
                    .simplify();
    access.scoped_exprs.emplace_back(expr);
  }

  return access;
}

std::function<float *(const std::vector<void *> &memory,
                      int indices[MAX_DEPTH])>
Compiler::gen_access_fn(const Compiler::Access &access,
                        LoopTree::TreeRef ref) const {
  const auto &expr = get_scoped_expr(access);
  const auto &constraints = get_constraints(access);
  auto resolve_expr =
      [&](const Expr &e) -> std::function<int64_t(int[MAX_DEPTH])> {
    auto sym_strides = get_symbol_strides(ref, access.alloc.lca);
    auto total_depth = lt.depth(ref);
    std::vector<int64_t> strides(total_depth, 0);
    for (const auto &sym : e.symbols()) {
      auto stride_expr = differentiate(e, sym).simplify();
      ASSERT(stride_expr.can_evaluate()) << "Invalid indexing expr";
      auto base_stride = stride_expr.evaluate();
      for (const auto &p : sym_strides.at(sym)) {
        const auto &sym_ref = p.first;
        const auto &sym_ref_stride = p.second;
        strides[lt.depth(sym_ref)] = sym_ref_stride * base_stride;
      }
    }
    auto offset_expr = intercept(expr);
    ASSERT(offset_expr.can_evaluate()) << "Invalid indexing expr";
    auto offset = offset_expr.evaluate();
    return [strides, offset](int indices[MAX_DEPTH]) -> int64_t {
      int64_t idx = 0;
      for (auto i = 0; i < strides.size(); ++i) {
        idx += indices[i] * strides[i];
      }
      return idx + offset;
    };
  };

  using IdxFn = std::function<int64_t(int[MAX_DEPTH])>;
  std::vector<std::pair<IdxFn, int64_t>> constraint_fns;
  for (const auto &c : constraints) {
    constraint_fns.emplace_back(resolve_expr(c.first), c.second);
  }
  auto idx_fn = resolve_expr(expr);
  auto mem_idx = access.alloc.mem_idx;
  return [constraint_fns, mem_idx, idx_fn](const std::vector<void *> &memory,
                                           int indices[MAX_DEPTH]) -> float * {
    for (const auto &p : constraint_fns) {
      auto i = p.first(indices);
      if (i < 0 || (p.second != -1 && i >= p.second)) {
        return nullptr;
      }
    }
    float *data = (float *)memory[mem_idx];
    return &data[idx_fn(indices)];
  };
}

InnerFnType Compiler::gen_mem_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  if (node.op() == Operation::read) {
    return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {};
  }

  ASSERT(node.inputs().size() == 1)
      << "Cannot call gen_mem_node on this node " << lt.ir.dump(node_ref);
  auto inacc = gen_access(node.inputs().at(0), ref);
  auto inacc_fn = gen_access_fn(inacc, ref);

  auto outacc = gen_access(node_ref, ref);
  auto outacc_fn = gen_access_fn(outacc, ref);

  auto s = lt.ir.dump(node_ref);
  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto *f = outacc_fn(memory, indices);
    if (!f) {
      return;
    }
    auto *i = inacc_fn(memory, indices);
    if (!i) {
      *f = 0;
    } else {
      *f = *i;
    }
  };
}

InnerFnType Compiler::gen_binary_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  using AccessFn =
      std::function<float *(const std::vector<void *> &, int[MAX_DEPTH])>;
  std::vector<AccessFn> inputs;
  for (const auto &inp : node.inputs()) {
    auto inacc = gen_access(inp, ref);
    auto inacc_fn = gen_access_fn(inacc, ref);
    inputs.emplace_back(inacc_fn);
  }

  auto outacc = gen_access(node_ref, ref);
  auto outacc_fn = gen_access_fn(outacc, ref);

  auto arith = [&]() -> std::function<float(float, float)> {
    switch (node.op()) {
      case Operation::add:
        return [=](float a, float b) -> float { return a + b; };
      case Operation::subtract:
        return [=](float a, float b) -> float { return a - b; };
      case Operation::multiply:
        return [=](float a, float b) -> float { return a * b; };
      case Operation::divide:
        return [=](float a, float b) -> float { return a / b; };
      case Operation::min:
        return [=](float a, float b) -> float { return a < b ? a : b; };
      case Operation::max:
        return [=](float a, float b) -> float { return a > b ? a : b; };
      default:
        ASSERT(0) << lt.ir.dump(node_ref) << " is not a binary operation";
    }
    return [=](float a, float b) -> float { return 0; };
  }();

  bool is_reduction = lt.ir.reduction_vars(node_ref).size();

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto *out_f = outacc_fn(memory, indices);
    if (!out_f) {
      return;
    }
    auto *f = inputs.at(0)(memory, indices);
    if (f && is_reduction) {
      *out_f = arith(*out_f, *f);
    } else if (f) {
      *out_f = *f;
    } else {
      *out_f = 0;
    }
    for (auto i = 1; i < inputs.size(); ++i) {
      auto other_inp = inputs.at(i);
      auto *f = other_inp(memory, indices);
      if (!f) {
        continue;
      }
      *out_f = arith(*out_f, *f);
    }
  };
}

InnerFnType Compiler::gen_unary_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  ASSERT(node.inputs().size() == 1);
  const auto &inp = node.inputs().at(0);

  auto inacc = gen_access(inp, ref);
  auto inacc_fn = gen_access_fn(inacc, ref);

  auto outacc = gen_access(node_ref, ref);
  auto outacc_fn = gen_access_fn(outacc, ref);

  auto arith = [&]() -> std::function<float(float)> {
    switch (node.op()) {
      case Operation::log:
        return [=](float a) -> float { return std::log(a); };
      case Operation::exp:
        return [=](float a) -> float { return std::exp(a); };
      case Operation::sqrt:
        return [=](float a) -> float { return std::sqrt(a); };
      case Operation::abs:
        return [=](float a) -> float { return std::abs(a); };
      case Operation::negate:
        return [=](float a) -> float { return -a; };
      case Operation::reciprocal:
        return [=](float a) -> float { return 1 / a; };
      default:
        ASSERT(0) << lt.ir.dump(node_ref) << " is not a unary operation";
    }
    return [=](float a) -> float { return 0; };
  }();

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto *f = outacc_fn(memory, indices);
    if (!f) {
      return;
    }
    auto *i = inacc_fn(memory, indices);
    if (!i) {
      *f = 0;
    } else {
      *f = arith(*i);
    }
  };
}

InnerFnType Compiler::gen_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);
  switch (node.op()) {
    case Operation::read:
    case Operation::view:
    case Operation::write:
      return gen_mem_node(ref);
    case Operation::add:
    case Operation::subtract:
    case Operation::multiply:
    case Operation::divide:
    case Operation::min:
    case Operation::max:
      return gen_binary_node(ref);
    case Operation::log:
    case Operation::exp:
    case Operation::sqrt:
    case Operation::reciprocal:
    case Operation::negate:
    case Operation::abs:
      return gen_unary_node(ref);
    default:
      ASSERT(0) << "Cannot generate node: " << lt.ir.dump(node_ref);
      return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
        ASSERT(0);
      };
  }
}

struct CPUInterpreted : public Compiled {
  std::vector<int64_t> intermediates;
  InnerFnType fn;
  std::string str;
  mutable std::vector<void *> mem;
  mutable std::vector<int64_t> mem_sizes;

  CPUInterpreted(const LoopTree &lt) {
    auto compiler = Compiler(lt);
    fn = compiler.gen_exec();
    str = compiler.gen_string();

    mem_sizes = compiler.memory_sizes();
    mem = allocate(mem_sizes);
  }

  ~CPUInterpreted() {
    for (auto i = 0; i < mem_sizes.size(); ++i) {
      if (mem_sizes[i] > 0) {
        free(mem[i]);
      }
    }
  }

  void run(const std::vector<void *> &memory, bool sync) const override {
    int indices[MAX_DEPTH] = {0};
    for (auto i = 0; i < memory.size(); ++i) {
      mem[i] = memory[i];
    }
    fn(mem, indices);
  }

  std::string dump() const override { return str; }
};

std::unique_ptr<Compiled> CPUInterpretedBackend::compile_impl(
    const LoopTree &lt) const {
  return std::make_unique<CPUInterpreted>(lt);
}

int CPUInterpretedBackend::hardware_requirement() const {
  // CPU is the only guaranteed hardware, always id = 0
  return 1 << 0;
}

static RegisterBackend cpu_backend_reg_(
    std::make_shared<CPUInterpretedBackend>());

}  // namespace loop_tool
