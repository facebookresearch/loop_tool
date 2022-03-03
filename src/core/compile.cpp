/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/compile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_set>

#include "loop_tool/backend.h"
#include "loop_tool/dynlib.h"
#include "loop_tool/error.h"
#include "loop_tool/symbolic.h"

namespace loop_tool {
using namespace symbolic;

using IdxFn = std::function<int64_t(int indices[MAX_DEPTH])>;

InnerFnType gen_fn(const LoopTree &lt, const Auxiliary &aux,
                   LoopTree::TreeRef ref, const GenFnType &callback);

// 0 -> not CPU
// 1 -> CPU
// 2 -> parallel CPU
int cpu_backend(const LoopTree &lt, LoopTree::TreeRef ref) {
  auto annot = lt.annotation(ref);
  if (annot == "parallel") {
    return 2;
  } else if (annot == "cpu") {
    return 1;
  } else if (annot == "") {  // default CPU
    return 1;
  }
  return 0;
}

// Return LCA of node and it's users
LoopTree::TreeRef get_scope(const LoopTree &lt, LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  const auto &node = lt.ir.node(lt.node(ref));

  // find all usage of this value
  std::unordered_set<IR::NodeRef> users = {node.outputs().begin(),
                                           node.outputs().end()};
  std::vector<LoopTree::TreeRef> uses;
  lt.walk([&](LoopTree::TreeRef tr, int) {
    if (lt.kind(tr) == LoopTree::NODE) {
      if (users.count(lt.node(tr))) {
        uses.emplace_back(tr);
      }
    }
  });
  ASSERT(users.size() == uses.size());

  // find least common ancestor
  auto ancestor = lt.parent(ref);
  if (node.op() == Operation::write) {
    ancestor = -1;
  } else {
    for (auto use : uses) {
      ancestor = lt.lca(lt.lca(ref, use), ancestor);
    }
  }

  return ancestor;
}

// check if there are reductions in the body of this tree
// over the variable
bool trivially_parallel(const LoopTree &lt, LoopTree::TreeRef ref) {
  bool threadable = true;
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
            threadable = false;
          }
        }
      },
      ref);
  return threadable;
}

/*
Nodes are values! Here's how memory is allocated:
1. Immediately deduce their scope - find LCA among dependents
2. Preserving the node's var order, find the minimum required memory to allocate
  for c:
    for a in 3r1: // shared scope
      for b in 2:
        for a in 5:
          create(T[a,b,c])
      for b in 2:
        for a in 5:
          for k: // note this is unrelated
            use(T[a,b,c])
  T[a, b, c], scope is {a:{5,0}, {b:{2,0}}, which means we need to allocate
       10 elements (scope for C and part of A is shared).  Even though
       A is a total of 16 elems, we can allocate 5 elems in the shared scope!
       We will partially use this allocation in the tail run (1/5th of it)
  We can also handle "weird" splits:
  for c:
    for a in 3r1: // shared scope
      for b in 2:
        for a in 2r1: // note the loops no longer match and theres a tail
          for a in 2:
            create(T[a,b,c])
      for b in 2:
        for a in 5:
          for k:
            use(T[a,b,c])
  Because we have strong invariants, we know that the LCA preserves sizing
  despite complex loop tiling.  we still have 5 a elems and 2 b elems for a
total of 10 The resultant expressions will share memory but will have different
  indexing semantics, which allows us to chain unique strategies without concern
  of accidental memory over-use

  gen_alloc maps a node to its LCA, size, idx into allocation map
  gen_idx_vec maps a user + alloc to an indexing vector
*/

void gen_alloc(const LoopTree &lt, Auxiliary &aux, LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  LoopTree::TreeRef lca = get_scope(lt, ref);
  auto alloc_idx = static_cast<int>(aux.allocs.size());

  // var -> running size, last tail
  auto loop_ref = lt.parent(ref);
  auto var_vec = lt.ir.node(lt.node(ref)).vars();
  std::unordered_set<IR::VarRef> vars = {var_vec.begin(), var_vec.end()};
  std::unordered_map<IR::VarRef, std::pair<int, int>> var_sizes;
  while (loop_ref != lca) {
    auto loop = lt.loop(loop_ref);
    auto size = loop.size;
    if (!vars.count(loop.var) || cpu_backend(lt, loop_ref) == 2) {
      loop_ref = lt.parent(loop_ref);
      continue;
    }
    if (var_sizes.count(loop.var)) {
      auto s = var_sizes[loop.var];
      size = size * (s.first + s.second);
    }
    var_sizes[loop.var] = std::make_pair(size, loop.tail);
    loop_ref = lt.parent(loop_ref);
  }

  int64_t total = 1;
  for (auto &p : var_sizes) {
    total *= (p.second.first + p.second.second);
  }

  // now we traverse upward and count the number of instances of this memory
  int64_t thread_multiplier = 1;
  loop_ref = lt.parent(ref);
  while (loop_ref != -1) {
    if (cpu_backend(lt, loop_ref) == 2) {
      auto loop = lt.loop(loop_ref);
      auto loop_size = loop.size + (loop.tail > 0);
      aux.thread_memory[loop_ref].emplace_back(
          std::make_pair(alloc_idx, thread_multiplier * total));
      thread_multiplier *= loop_size;
    }
    loop_ref = lt.parent(loop_ref);
  }

  auto node_ref = lt.node(ref);
  bool reduction = (lt.ir.pointwise_vars(node_ref).size() !=
                    lt.ir.loop_vars(node_ref).size());
  bool should_init = false;
  float init_val = -1337;
  if (lt.ir.node(node_ref).op() == Operation::multiply) {
    should_init = reduction;
    init_val = 1;
  } else if (lt.ir.node(node_ref).op() == Operation::add) {
    should_init = reduction;
    init_val = 0;
  }

  Allocation alloc{
      total, thread_multiplier, alloc_idx, should_init, init_val, lca, ref};
  aux.allocs[node_ref] = alloc;
  aux.resets[alloc.lca].emplace_back(alloc);
}

std::vector<std::pair<int, int64_t>> gen_idx_vector(const LoopTree &lt,
                                                    const Auxiliary &aux,
                                                    const Allocation &alloc,
                                                    LoopTree::TreeRef use) {
  std::vector<std::pair<int, int64_t>> idx_vec;
  auto user_ref = lt.node(use);
  auto producer_ref = lt.node(alloc.producer);

  // get index of loop into indices[MAX_DEPTH]
  // by counting number of parents
  auto user_loop_ref = lt.parent(use);
  if (user_loop_ref == -1) {
    return idx_vec;
  }
  auto depth = lt.tree_node(user_loop_ref).depth;

  auto producer = lt.ir.node(producer_ref);
  auto user = lt.ir.node(user_ref);
  auto producer_vars = producer.vars();
  auto user_vars = lt.ir.loop_vars(user_ref);
  std::unordered_set<IR::VarRef> user_vars_set = {user_vars.begin(),
                                                  user_vars.end()};
  std::unordered_set<IR::VarRef> producer_vars_set = {producer_vars.begin(),
                                                      producer_vars.end()};

  // virtual var -> expr on var, original var
  std::unordered_map<IR::VarRef, std::pair<symbolic::Expr, IR::VarRef>>
      user_view_vars;
  std::unordered_map<IR::VarRef, std::vector<IR::VarRef>> mapped_view_vars;

  // We're used by a view, meaning we have to find vars that are implicitly
  // used. e.g. read(X) <- view(X, A + B) will have loops A, B but implicitly
  // depends on X so we track loops A, B and then map them into var X.
  // user_view_vars will keep this information, mapping A -> {X, A+B}, B -> {X,
  // A+B}
  if (user.op() == Operation::view && (producer_ref != user_ref)) {
    for (const auto &c : user.constraints()) {
      if (c.first.type() != Expr::Type::symbol) {
        continue;
      }
      std::vector<symbolic::Symbol> view_symbols;
      auto collect_vars = [&](symbolic::Expr e) {
        if (e.type() == symbolic::Expr::Type::symbol) {
          view_symbols.emplace_back(e.symbol());
        }
        return e;
      };
      c.second.walk(collect_vars);
      auto sym = c.first.symbol();
      ASSERT(user.has_sym(sym))
          << "cannot find var for symbolic constraint on " << c.first.dump();
      IR::VarRef orig_var = user.var(sym);
      // we can't get information from mapping a user var to another user var
      if (user_vars_set.count(orig_var)) {
        continue;
      }
      for (auto sym : view_symbols) {
        if (user.has_sym(sym)) {
          auto v = user.var(sym);
          auto stride = differentiate(c.second, sym);
          ASSERT(user_view_vars.count(v) == 0)
              << "mapping already mapped variable " << lt.ir.var(v).name()
              << " (sym: " << symbolic::Expr(sym).dump() << ")";
          user_view_vars.insert(
              std::make_pair(v, std::make_pair(stride, orig_var)));
          mapped_view_vars[orig_var].emplace_back(v);
        }
      }
    }
  }

  // first we collect the orders of each var
  std::unordered_map<IR::VarRef, std::vector<LoopTree::TreeRef>>
      producer_var_loops;
  std::unordered_map<IR::VarRef, std::vector<LoopTree::TreeRef>> user_var_loops;
  while (user_loop_ref != alloc.lca) {
    auto loop = lt.loop(user_loop_ref);
    if (aux.thread_memory.count(user_loop_ref)) {
      // handled by threading!
    } else if (producer_vars_set.count(loop.var) ||
               user_view_vars.count(loop.var)) {
      producer_var_loops[loop.var].emplace_back(user_loop_ref);
    }
    user_loop_ref = lt.parent(user_loop_ref);
  }
  std::reverse(producer_vars.begin(), producer_vars.end());
  // producer has real layout {user_vars}, user (if view) has virtual layout
  auto inner_size_for_var = [&](IR::VarRef v) {
    auto inner = [&](IR::VarRef v) {
      int64_t size = 1;
      for (auto iv : producer_vars) {
        if (iv == v) {
          break;
        }
        if (!producer_var_loops.count(iv)) {
          continue;
        }
        int64_t var_size = 1;
        for (auto l : producer_var_loops.at(iv)) {
          auto loop = lt.loop(l);
          var_size = loop.size * var_size + loop.tail;
        }
        size *= var_size;
      }
      return size;
    };
    if (producer_vars_set.count(v)) {
      return inner(v);
    }
    // the idea here is that the user can use a *different* variable
    // than one of the ones the producer has.
    // We simply map these different ones to the producer equivalent
    ASSERT(user_view_vars.count(v));
    auto base_var = user_view_vars.at(v).second;
    return inner(base_var);
  };
  for (const auto &v : producer_vars) {
    auto inner_size = inner_size_for_var(v);
    // could be omitted due to LCA or it could be a view mapping
    if (!producer_var_loops.count(v) && !mapped_view_vars.count(v)) {
      continue;
    }
    if (mapped_view_vars.count(v)) {
      ASSERT(user.op() == Operation::view);
      for (auto mv : mapped_view_vars.at(v)) {
        ASSERT(user_view_vars.count(mv));
        auto stride_override = user_view_vars.at(mv);
        auto inner_scale = stride_override.first;
        auto orig_var = stride_override.second;
        ASSERT(inner_scale.type() == symbolic::Expr::Type::value)
            << "cannot handle symbolic stride (yet): " << inner_scale.dump();
        auto size = inner_size * inner_scale.value();
        for (auto l : producer_var_loops.at(mv)) {
          auto idx = lt.tree_node(l).depth;
          idx_vec.emplace_back(std::make_pair(idx, size));
          auto loop = lt.loop(l);
          size = loop.size * size + loop.tail * inner_size;
        }
      }
      continue;
    }

    auto size = inner_size;
    for (auto l : producer_var_loops.at(v)) {
      auto idx = lt.tree_node(l).depth;
      idx_vec.emplace_back(std::make_pair(idx, size));
      auto loop = lt.loop(l);
      size = loop.size * size + loop.tail * inner_size;
    }
  }
  return idx_vec;
}

/*
 an index function maps a point in memory given a location in the loop tree
*/
std::function<int64_t(int[MAX_DEPTH])> gen_idx_func(const LoopTree &lt,
                                                    const Auxiliary &aux,
                                                    const Allocation &alloc,
                                                    LoopTree::TreeRef use) {
  auto ref = alloc.producer;
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  ASSERT(lt.kind(use) == LoopTree::NODE);

  auto idx_vec = gen_idx_vector(lt, aux, alloc, use);
  return [=](int indices[MAX_DEPTH]) {
    int64_t idx = 0;
    for (const auto &p : idx_vec) {
      idx += indices[p.first] * p.second;
    }
    return idx;
  };
}

InnerFnType gen_read(const LoopTree &lt, const Auxiliary &aux,
                     LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  const Allocation &alloc = aux.allocs.at(lt.node(ref));
  int external_memory = -1;
  for (auto i = 0; i < lt.ir.inputs().size(); ++i) {
    if (lt.ir.inputs()[i] == lt.node(ref)) {
      external_memory = i;
    }
  }
  ASSERT(external_memory > -1 && "No input found!");
  auto idx_fn = gen_idx_func(lt, aux, alloc, ref);
  auto alloc_read = alloc;
  // TODO this is a hacky way to ensure all variables are in the indexing
  alloc_read.lca = -1;
  auto saved_threading = aux.thread_memory;
  const_cast<Auxiliary &>(aux).thread_memory.clear();
  auto read_idx_fn = gen_idx_func(lt, aux, alloc_read, ref);
  const_cast<Auxiliary &>(aux).thread_memory = saved_threading;
  auto inp_memory = alloc.idx + lt.ir.inputs().size() + lt.ir.outputs().size();

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    for (auto i = 0; i < MAX_DEPTH; ++i) {
      if (tails[i]) {
        return;
      }
    }
    auto to_idx = idx_fn(indices);
    auto from_idx = read_idx_fn(indices);
    ((float *)memory[inp_memory])[to_idx] =
        ((float *)memory[external_memory])[from_idx];
  };
}

InnerFnType gen_write(const LoopTree &lt, const Auxiliary &aux,
                      LoopTree::TreeRef ref) {
  int external_memory = -1;
  auto tree_node = lt.tree_node(ref);
  for (auto i = 0; i < lt.ir.outputs().size(); ++i) {
    if (lt.ir.outputs()[i] == tree_node.node) {
      external_memory = i + lt.ir.inputs().size();
    }
  }
  ASSERT(external_memory > -1 && "No output found!");

  const auto &n = lt.ir.node(tree_node.node);
  ASSERT(n.inputs().size() == 1);
  ASSERT(n.outputs().size() == 0);

  auto inp = n.inputs()[0];

  auto inp_idx_fn = gen_idx_func(lt, aux, aux.allocs.at(inp), ref);
  auto out_idx_fn = gen_idx_func(lt, aux, aux.allocs.at(tree_node.node), ref);
  auto alloc = aux.allocs.at(tree_node.node);
  auto input_memory =
      aux.allocs.at(inp).idx + lt.ir.inputs().size() + lt.ir.outputs().size();

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    for (auto i = 0; i < MAX_DEPTH; ++i) {
      if (tails[i]) {
        return;
      }
    }
    ((float *)memory[external_memory])[out_idx_fn(indices)] =
        ((float *)memory[input_memory])[inp_idx_fn(indices)];
  };
}

InnerFnType gen_add(const LoopTree &lt, const Auxiliary &aux,
                    LoopTree::TreeRef ref) {
  auto tree_node = lt.tree_node(ref);
  const auto &n = lt.ir.node(tree_node.node);

  std::vector<std::pair<std::function<int64_t(int[MAX_DEPTH])>, int>> inputs;
  std::pair<std::function<int64_t(int[MAX_DEPTH])>, int> output;

  auto mem_off = lt.ir.inputs().size() + lt.ir.outputs().size();
  for (auto &inp_ref : n.inputs()) {
    const auto &alloc = aux.allocs.at(inp_ref);
    inputs.emplace_back(gen_idx_func(lt, aux, alloc, ref), alloc.idx + mem_off);
  }
  auto out_alloc = aux.allocs.at(tree_node.node);

  output = std::make_pair(gen_idx_func(lt, aux, out_alloc, ref),
                          out_alloc.idx + mem_off);
  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    for (auto i = 0; i < MAX_DEPTH; ++i) {
      if (tails[i]) {
        return;
      }
    }
    for (auto inp : inputs) {
      ((float *)memory[output.second])[output.first(indices)] +=
          ((float *)memory[inp.second])[inp.first(indices)];
    }
  };
};

InnerFnType gen_mul(const LoopTree &lt, const Auxiliary &aux,
                    LoopTree::TreeRef ref) {
  auto tree_node = lt.tree_node(ref);
  const auto &n = lt.ir.node(tree_node.node);

  std::vector<std::pair<std::function<int64_t(int[MAX_DEPTH])>, int>> inputs;
  std::pair<std::function<int64_t(int[MAX_DEPTH])>, int> output;

  auto mem_off = lt.ir.inputs().size() + lt.ir.outputs().size();
  for (auto &inp_ref : n.inputs()) {
    const auto &alloc = aux.allocs.at(inp_ref);
    inputs.emplace_back(gen_idx_func(lt, aux, alloc, ref), alloc.idx + mem_off);
  }
  auto out_alloc = aux.allocs.at(tree_node.node);

  output = std::make_pair(gen_idx_func(lt, aux, out_alloc, ref),
                          out_alloc.idx + mem_off);
  auto depth = lt.tree_node(ref).depth;
  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    for (auto i = 0; i < MAX_DEPTH; ++i) {
      if (tails[i]) {
        return;
      }
    }
    for (auto inp : inputs) {
      ((float *)memory[output.second])[output.first(indices)] *=
          ((float *)memory[inp.second])[inp.first(indices)];
    }
  };
};

InnerFnType gen_view(const LoopTree &lt, const Auxiliary &aux,
                     LoopTree::TreeRef ref) {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  const Allocation &alloc = aux.allocs.at(lt.node(ref));

  const auto &node = lt.ir.node(lt.node(ref));
  ASSERT(node.inputs().size() == 1) << "Cannot execute multi input views yet";
  const auto &dep = node.inputs().at(0);
  auto &dep_alloc = aux.allocs.at(dep);
  auto dep_memory_idx =
      dep_alloc.idx + lt.ir.inputs().size() + lt.ir.outputs().size();
  auto memory_idx = alloc.idx + lt.ir.inputs().size() + lt.ir.outputs().size();

  auto dep_idx_fn = gen_idx_func(lt, aux, dep_alloc, ref);
  auto idx_fn = gen_idx_func(lt, aux, alloc, ref);

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    for (auto i = 0; i < MAX_DEPTH; ++i) {
      if (tails[i]) {
        return;
      }
    }
    ((float *)memory[memory_idx])[idx_fn(indices)] =
        ((float *)memory[dep_memory_idx])[dep_idx_fn(indices)];
  };
}

InnerFnType gen_leaf(const LoopTree &lt, const Auxiliary &aux,
                     LoopTree::TreeRef ref) {
  auto tree_node = lt.tree_node(ref);
  const auto &n = lt.ir.node(tree_node.node);

  auto alloc = aux.allocs.at(lt.node(ref));

  if (n.op() == Operation::add) {
    return gen_add(lt, aux, ref);
  } else if (n.op() == Operation::multiply) {
    return gen_mul(lt, aux, ref);
  } else if (n.op() == Operation::read) {
    return gen_read(lt, aux, ref);
  } else if (n.op() == Operation::write) {
    return gen_write(lt, aux, ref);
  } else if (n.op() == Operation::view) {
    return gen_view(lt, aux, ref);
  }
  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    ASSERT(0) << "Cannot execute operation " << loop_tool::dump(n.op())
              << " in\n"
              << lt.dump();
    return;
  };
}

std::function<void(const std::vector<void *> &)> gen_mem(
    const LoopTree &lt, const Auxiliary &aux, LoopTree::TreeRef ref) {
  // does this loop need to reset anything?
  std::vector<Allocation> reset_allocs =
      aux.resets.count(ref) ? aux.resets.at(ref) : std::vector<Allocation>{};
  auto alloc_off = lt.ir.inputs().size() + lt.ir.outputs().size();

  return [=](const std::vector<void *> &memory) {
    for (auto alloc : reset_allocs) {
      auto idx = alloc.idx + alloc_off;
      if (alloc.init_val == 0) {
        memset(memory[idx], 0, sizeof(float) * alloc.size);
      } else {
        for (auto i = 0; i < alloc.size; ++i) {
          ((float *)memory[idx])[i] = alloc.init_val;
        }
      }
    }
  };
}

InnerFnType gen_parallel_loop(const LoopTree &lt, const Auxiliary &aux,
                              LoopTree::TreeRef ref,
                              const GenFnType &callback) {
  auto tree_node = lt.tree_node(ref);
  auto depth = tree_node.depth;
  auto loop = tree_node.loop;
  auto size = loop.size;
  auto tail_size = loop.tail;
  auto var_idx = aux.var_idx.at(loop.var);

  ASSERT(size > 0);
  ASSERT(tail_size >= 0);
  std::vector<InnerFnType> fns;
  for (auto c : tree_node.children) {
    fns.emplace_back(gen_fn(lt, aux, c, callback));
  }

  auto inner_size = aux.inner_size.at(ref);
  auto memory_fn = gen_mem(lt, aux, ref);

  // to handle threading, we calculate offsets memory into memory
  // for (auto& mem : aux.threading.at(ref)) {
  //  //mem.idx
  //}
  auto alloc_off = lt.ir.inputs().size() + lt.ir.outputs().size();
  auto offset_memory = [=](const std::vector<void *> &memory_, int i) {
    auto memory = memory_;
    if (!aux.thread_memory.count(ref)) {
      return memory;
    }
    // some memory is threaded, we have to
    // 1. find that memory
    // 2. find how that thread strides the memory
    // 3. mutate the memory as `address = (address + i * stride)`
    // this means we need TreeRef -> { (idx, stride), (idx, stride) }
    for (auto &p : aux.thread_memory.at(ref)) {
      auto mem_idx = alloc_off + p.first;
      auto fmem = (float *)(memory[mem_idx]);
      memory[mem_idx] = fmem + i * p.second;
    }
    return memory;
  };

  return [=](const std::vector<void *> &memory_, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    auto run = [&](int n_size, int t_size) {
      std::vector<std::thread> threads;
      for (auto i = 0; i < n_size; ++i) {
        auto memory = offset_memory(memory_, i);
        threads.emplace_back([=]() {
          int indices_[MAX_DEPTH];
          std::copy(indices, indices + MAX_DEPTH, indices_);
          int tails_[MAX_DEPTH];
          std::copy(tails, tails + MAX_DEPTH, tails_);
          memory_fn(memory);
          for (const auto &fn : fns) {
            indices_[depth] = i;
            tails_[var_idx] = 0;
            fn(memory, indices_, tails_);
          }
        });
      }
      for (auto &t : threads) {
        t.join();
      }
      if (t_size) {
        auto memory = offset_memory(memory_, n_size);
        memory_fn(memory);
        for (const auto &fn : fns) {
          indices[depth] = n_size;
          tails[var_idx] = t_size;
          fn(memory, indices, tails);
        }
      }
    };

    auto tail = tails[var_idx];
    if (tail) {
      auto N = tail / inner_size;
      auto T = tail % inner_size;
      run(N, T);
      return;
    }

    run(size, tail_size);
  };
}

InnerFnType gen_loop(const LoopTree &lt, const Auxiliary &aux,
                     LoopTree::TreeRef ref, const GenFnType &callback) {
  auto backend = cpu_backend(lt, ref);
  ASSERT(backend) << "backend not yet implemented: " << lt.annotation(ref);
  if (backend == 2) {
    ASSERT(trivially_parallel(lt, ref))
        << "threaded reductions not yet supported";
    return gen_parallel_loop(lt, aux, ref, callback);
  }
  auto tree_node = lt.tree_node(ref);
  auto depth = tree_node.depth;
  auto loop = tree_node.loop;
  auto size = loop.size;
  auto tail_size = loop.tail;
  auto var_idx = aux.var_idx.at(loop.var);

  ASSERT(size > 0) << "invalid size for loop L" << ref << " in\n" << lt.dump();
  ASSERT(tail_size >= 0);
  std::vector<InnerFnType> fns;
  for (auto c : tree_node.children) {
    fns.emplace_back(gen_fn(lt, aux, c, callback));
  }

  auto inner_size = aux.inner_size.at(ref);
  auto memory_fn = gen_mem(lt, aux, ref);

  ASSERT(depth < MAX_DEPTH);
  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    auto run = [&](int n_size, int t_size) {
      for (auto i = 0; i < n_size; ++i) {
        memory_fn(memory);
        for (const auto &fn : fns) {
          indices[depth] = i;
          tails[var_idx] = 0;
          fn(memory, indices, tails);
        }
      }
      if (t_size) {
        memory_fn(memory);
        for (const auto &fn : fns) {
          indices[depth] = n_size;
          tails[var_idx] = t_size;
          fn(memory, indices, tails);
        }
      }
    };

    auto tail = tails[var_idx];
    if (tail) {
      auto N = tail / inner_size;
      auto T = tail % inner_size;
      run(N, T);
      return;
    }

    run(size, tail_size);
  };
}

void update_inner_size(
    const LoopTree &lt,
    std::unordered_map<LoopTree::TreeRef, int64_t> &inner_size,
    LoopTree::TreeRef ref) {
  // can only be done with leaf nodes
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  auto loop_ref = lt.parent(ref);
  std::unordered_map<IR::VarRef, std::pair<int, int>> var_sizes;
  while (loop_ref != -1) {
    auto loop = lt.loop(loop_ref);
    if (!var_sizes.count(loop.var)) {
      var_sizes[loop.var] = std::make_pair(1, 0);
    }
    auto s = var_sizes[loop.var];
    auto old_size = s.first + s.second;
    if (inner_size.count(loop_ref)) {
      ASSERT(inner_size[loop_ref] == old_size)
          << "found inner size " << inner_size[loop_ref] << " but expected "
          << old_size << "\n"
          << lt.dump();
    } else {
      inner_size[loop_ref] = old_size;
    }
    auto new_size = loop.size * old_size;
    var_sizes[loop.var] = std::make_pair(new_size, loop.tail);
    loop_ref = lt.parent(loop_ref);
  }
}

InnerFnType gen_fn(const LoopTree &lt, const Auxiliary &aux,
                   LoopTree::TreeRef ref, const GenFnType &callback) {
  ASSERT(lt.tree_node(ref).depth < MAX_DEPTH);
  if (callback) {
    auto callback_fn = callback(lt, aux, ref);
    if (callback_fn) {
      return callback_fn;
    }
  }
  if (lt.kind(ref) == LoopTree::NODE) {
    return gen_leaf(lt, aux, ref);
  } else {
    return gen_loop(lt, aux, ref, callback);
  }
}

// recursively calculate all auxilary information
void gen_aux(const LoopTree &lt, Auxiliary &aux, LoopTree::TreeRef ref) {
  ASSERT(lt.tree_node(ref).depth < MAX_DEPTH);
  if (lt.kind(ref) == LoopTree::NODE) {
    update_inner_size(lt, aux.inner_size, ref);
    gen_alloc(lt, aux, ref);
  } else {
    auto loop = lt.loop(ref);
    if (!aux.var_idx.count(loop.var)) {
      auto idx = aux.var_idx.size();
      aux.var_idx[loop.var] = idx;
    }
    for (auto c : lt.tree_node(ref).children) {
      gen_aux(lt, aux, c);
    }
  }
}

Auxiliary calculate_aux(const LoopTree &lt) {
  Auxiliary aux;
  for (auto root : lt.roots) {
    gen_aux(lt, aux, root);
  }
  return aux;
}

// function + sizes for intermediates
std::pair<std::function<void(const std::vector<void *> &)>,
          std::vector<int64_t>>
compile(const LoopTree &lt,
        std::function<InnerFnType(const LoopTree &, const Auxiliary &,
                                  LoopTree::TreeRef)>
            callback) {
  Auxiliary aux = calculate_aux(lt);

  std::vector<InnerFnType> fns;
  for (auto root : lt.roots) {
    fns.emplace_back(gen_fn(lt, aux, root, callback));
  }
  auto memory_fn = gen_mem(lt, aux, -1);

  auto fn = [=](const std::vector<void *> &memory) {
    // TODO this is slow
    memory_fn(memory);
    for (const auto &fn : fns) {
      int indices[MAX_DEPTH] = {0};
      int tails[MAX_DEPTH] = {0};
      fn(memory, indices, tails);
    }
  };

  std::vector<int64_t> sizes;
  sizes.resize(aux.allocs.size());
  for (const auto &p : aux.allocs) {
    auto sizes_idx = p.second.idx;
    ASSERT(sizes.size() > sizes_idx);
    ASSERT(sizes_idx > -1);
    sizes[sizes_idx] = p.second.size * sizeof(float) * p.second.thread_size;
  }
  return std::make_pair(fn, sizes);
};

void exec(const LoopTree &lt, const std::vector<void *> &memory) {
  auto p = compile(lt);
  auto c = p.first;
  auto memory_w_intermediates = memory;
  std::vector<void *> free_me;
  for (auto s : p.second) {
    memory_w_intermediates.emplace_back(calloc(1, s));
    free_me.emplace_back(memory_w_intermediates.back());
  }

  c(memory_w_intermediates);
  for (auto v : free_me) {
    free(v);
  }
}

size_t getCount() {
  static size_t count = 0;
  return count++;
}

Compiler::Compiler(const LoopTree &lt_) : lt(lt_) {
  count = getCount();
  std::vector<LoopTree::TreeRef> reverse_order;
  lt.walk([&](LoopTree::TreeRef ref, int) { reverse_order.emplace_back(ref); });
  std::reverse(reverse_order.begin(), reverse_order.end());

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
        // ASSERT(var_sizes.at(p.first) == tmp_sizes.at(p.first))
        //    << "incompatible loop sizes for " << lt.ir.var(p.first).name()
        //    << " found " << tmp_sizes.at(p.first) << " expected "
        //    << var_sizes.at(p.first);
        // continue;
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

  for (auto &ref : reverse_order) {
    if (lt.kind(ref) == LoopTree::NODE) {
      cur_sizes.clear();
      continue;
    }
    auto loop = lt.loop(ref);
    auto inner_size = cur_sizes.count(loop.var) ? cur_sizes.at(loop.var) : 1;
    if (inner_sizes.count(ref)) {
      ASSERT(inner_sizes.at(ref) == inner_size);
      continue;
    }

    inner_sizes[ref] = inner_size;
    int64_t var_size = loop.size * inner_size + loop.tail;
    cur_sizes[loop.var] = var_size;
    if (var_sizes.count(loop.var)) {
      var_size = std::max(var_sizes.at(loop.var), var_size);
    }
    var_sizes[loop.var] = var_size;
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
InnerFnTypeImproved Compiler::gen_loop(
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

  std::vector<InnerFnTypeImproved> body_children;
  std::vector<InnerFnTypeImproved> tail_children;
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

InnerFnTypeImproved Compiler::gen_reset(LoopTree::TreeRef ref) const {
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

InnerFnTypeImproved Compiler::gen_exec(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  if (ref == -1) {
    std::vector<InnerFnTypeImproved> roots;
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
  if (lt.kind(ref) == LoopTree::NODE) {
    return gen_node(ref);
  }
  ASSERT(lt.kind(ref) == LoopTree::LOOP);
  return gen_loop(ref, overrides);
}

bool Compiler::is_input_output(IR::NodeRef nr) const {
  for (auto i : lt.ir.inputs()) {
    if (nr == i) {
      return true;
    }
  }
  for (auto o : lt.ir.outputs()) {
    if (nr == o) {
      return true;
    }
  }
  return false;
};

std::string Compiler::gen_access_string(IR::NodeRef node_ref,
                                        LoopTree::TreeRef ref) const {
  std::stringstream ss;
  auto acc = gen_access(node_ref, ref);
  std::unordered_map<Symbol, std::string, Hash<Symbol>> sym_strings;
  auto p = lt.parent(ref);
  while (p != acc.alloc.lca) {
    const auto &l = lt.loop(p);
    auto sym = var_to_sym.at(l.var);
    std::stringstream sym_str;
    if (sym_strings.count(sym)) {
      sym_str << sym_strings.at(sym) << "+";
    }
    sym_str << "i_" << std::to_string(p);
    auto stride = inner_sizes.at(p);
    if (stride > 1) {
      sym_str << "*" << stride;
    }
    sym_strings[var_to_sym.at(l.var)] = sym_str.str();
    p = lt.parent(p);
  }
  for (const auto &p : sym_strings) {
    sym_strings[p.first] = "(" + p.second + ")";
  }

  bool constrained = false;
  for (const auto &b : acc.bounds) {
    if (b.first != 0 || b.second != -1) {
      constrained = true;
    }
  }
  // TODO eliminate this restriction
  if (node_ref == lt.node(ref)) {
    constrained = false;
  }

  if (constrained) {
    ss << "(";
    bool constraint_added = false;
    for (auto i = 0; i < acc.bounds.size(); ++i) {
      const auto &b = acc.bounds.at(i);
      const auto &expr = acc.scoped_exprs.at(i);
      if (b.first == 0 && b.second == -1) {
        continue;
      }
      const auto &str = expr.dump(false, sym_strings);
      if (constraint_added) {
        ss << " && ";
      }
      ss << "((" << str << ") >= 0)";
      if (b.second != -1) {
        ss << " && ((" << str << ") < " << b.second << ")";
      }
      constraint_added = true;
    }
    ss << " ? ";
  }

  if (acc.alloc.size() > 1 || is_input_output(acc.alloc.node_ref)) {
    ss << "((float*)memory[" << acc.alloc.mem_idx << "])";
    ss << "[";
    for (auto i = 0; i < acc.scoped_exprs.size(); ++i) {
      if (!acc.alloc.sizes.at(i)) {
        continue;
      }
      const auto &expr = acc.scoped_exprs.at(i);
      ss << (expr * Expr(acc.alloc.size(i + 1))).dump(false, sym_strings);
      if (i != acc.scoped_exprs.size() - 1) {
        ss << "+";
      }
    }
    if (acc.scoped_exprs.size() == 0) {
      ss << "0";
    }
    ss << "]";
  } else {
    ss << "v" << acc.alloc.mem_idx;
  }

  if (constrained) {
    ss << " : " << 0 << ")";
  }
  return ss.str();
}

std::string Compiler::gen_mem_node_string(LoopTree::TreeRef ref) const {
  std::stringstream ss;
  const auto &node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);
  ASSERT(node.inputs().size() == 1);
  ss << gen_access_string(node_ref, ref);
  ss << " = ";
  ss << gen_access_string(node.inputs().at(0), ref);
  ss << ";";
  return ss.str();
}

std::string Compiler::gen_reset_string(LoopTree::TreeRef ref) const {
  std::stringstream ss;
  auto line_prefix = gen_indent(ref, 1);
  auto value = [&](const Node &node) -> float {
    if (node.op() == Operation::add) {
      return 0;
    } else if (node.op() == Operation::multiply) {
      return 1;
    } else if (node.op() == Operation::min) {
      return std::numeric_limits<float>::max();
    } else if (node.op() == Operation::max) {
      return -std::numeric_limits<float>::max();
    } else if (node.op() == Operation::write) {
      return 0;  // TODO fix
    } else if (node.op() == Operation::view) {
      return 0;  // TODO fix
    }
    ASSERT(0) << "cannot find default value for " << dump(node.op());
    return -1;
  };
  for (const auto &p : allocations) {
    const auto &alloc = p.second;
    if (alloc.lca == ref) {
      const auto &node = lt.ir.node(alloc.node_ref);
      bool needs_set = lt.ir.reduction_vars(alloc.node_ref).size() &&
                       node.op() != Operation::view;
      for (const auto &input : node.inputs()) {
        if (lt.ir.node(input).op() == Operation::view &&
            !lt.scheduled.count(input)) {
          needs_set = true;
        }
      }
      if (!lt.scheduled.count(alloc.node_ref)) {
        continue;
      } else if (alloc.size() == 1 && !(is_input_output(alloc.node_ref))) {
        ss << line_prefix << "float v" << alloc.mem_idx;
        if (needs_set) {
          ss << " = " << value(node);
        }
        ss << ";\n";
      } else if (needs_set) {
        set_called = true;
        ss << line_prefix << "set((float*)memory[" << alloc.mem_idx << "], ";
        ss << value(node) << ", " << alloc.size() << ");\n";
      }
    }
  }
  return ss.str();
}

std::string Compiler::gen_compute_node_string(LoopTree::TreeRef ref) const {
  std::stringstream ss;
  const auto &node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  bool is_infix = [&]() {
    switch (node.op()) {
      case Operation::add:
      case Operation::multiply:
      case Operation::subtract:
      case Operation::divide:
        return true;
      default:
        return false;
    }
  }();
  bool is_binary = [&]() {
    switch (node.op()) {
      case Operation::add:
      case Operation::multiply:
      case Operation::subtract:
      case Operation::divide:
      case Operation::min:
      case Operation::max:
        return true;
      default:
        return false;
    }
  }();
  auto op = [&]() {
    switch (node.op()) {
      case Operation::add:
        return "+";
      case Operation::multiply:
        return "*";
      case Operation::subtract:
        return "-";
      case Operation::divide:
        return "/";
      case Operation::max:
        return "max";
      case Operation::min:
        return "min";
      case Operation::log:
        return "log";
      case Operation::exp:
        return "exp";
      case Operation::sqrt:
        return "sqrt";
      case Operation::negate:
        return "-";
      case Operation::abs:
        return "abs";
      case Operation::reciprocal:
        return "1 / ";
      default:
        ASSERT(0) << "can't emit code for " << dump(node.op());
        return "";
    }
  }();

  ss << gen_access_string(node_ref, ref);
  ss << " = ";

  bool is_reduction = lt.ir.reduction_vars(node_ref).size();
  std::vector<std::string> access_strings;
  if (is_reduction) {
    access_strings.emplace_back(gen_access_string(node_ref, ref));
  }
  for (const auto &inp : node.inputs()) {
    access_strings.emplace_back(gen_access_string(inp, ref));
  }

  if (is_infix) {
    for (const auto &access_string : access_strings) {
      ss << access_string;
      if (&access_string != &access_strings.back()) {
        ss << " " << op << " ";
      }
    }
  } else if (is_binary) {
    std::function<void(int)> nest;
    nest = [&](int i) {
      if (i == access_strings.size() - 1) {
        ss << access_strings.at(i);
        return;
      }
      ss << op << "(" << access_strings.at(i) << ", ";
      nest(i + 1);
      ss << ")";
    };
    nest(0);
  } else {
    ASSERT(access_strings.size() == 1);
    ss << op << "(" << access_strings.at(0) << ")";
  }
  ss << ";";
  return ss.str();
}

std::string Compiler::gen_node_string(LoopTree::TreeRef ref) const {
  std::stringstream ss;
  auto line_prefix = gen_indent(ref);
  const auto &node = lt.ir.node(lt.node(ref));

  if (lt.children(lt.parent(ref)).at(0) == ref) {
    ss << gen_reset_string(lt.parent(ref));
  }
  ss << line_prefix;
  switch (node.op()) {
    case Operation::write:
    case Operation::view:
      ss << gen_mem_node_string(ref);
      break;
    case Operation::read:
      break;
    default:
      ss << gen_compute_node_string(ref);
  }
  ss << " // %" << lt.node(ref) << " (" << dump(node.op()) << ")\n";
  return ss.str();
}

std::string Compiler::gen_loop_string(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  std::stringstream ss;
  auto line_prefix = gen_indent(ref);

  const auto &loop = lt.loop(ref);
  std::string iter_var = "i_" + std::to_string(ref);

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

  std::vector<std::string> body_children;
  std::vector<std::string> tail_children;
  for (auto c : lt.children(ref)) {
    body_children.emplace_back(gen_string(c, overrides));
  }
  if (tail > 0) {
    // find first loop of same var, and override
    overrides[loop.var] = tail;
    for (const auto &cref : lt.children(ref)) {
      tail_children.emplace_back(gen_string(cref, overrides));
    }
  }

  if (lt.children(lt.parent(ref)).at(0) == ref) {
    ss << gen_reset_string(lt.parent(ref));
  }
  ss << line_prefix << "for (int64_t " << iter_var << " = 0L; ";
  ss << iter_var << " < " << size << "L; ++" << iter_var << ") { // "
     << lt.ir.var(loop.var).name() << "\n";
  for (auto c : body_children) {
    ss << c;
  }
  ss << line_prefix << "}\n";
  if (tail > 0) {
    ss << line_prefix << "{ // " << lt.ir.var(loop.var).name() << " tail\n";
    ss << gen_indent(ref, 1) << "int64_t " << iter_var << " = " << loop.size
       << "L;\n";
    for (auto c : tail_children) {
      ss << c;
    }
    ss << line_prefix << "}\n";
  }
  return ss.str();
}

std::string Compiler::gen_string(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  if (ref == -1) {
    // generate the body first to minimize header code
    std::stringstream body;
    for (auto c : lt.roots) {
      body << gen_string(c);
    }
    std::stringstream ss;
    bool define_max = false;
    bool define_min = false;
    for (auto n : lt.ir.nodes()) {
      if (lt.ir.node(n).op() == Operation::max) {
        define_max = true;
      }
      if (lt.ir.node(n).op() == Operation::min) {
        define_min = true;
      }
    }

    ss << R"""(#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

)""";
    if (define_max) {
      ss << R"""(
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
)""";
    }
    if (define_min) {
      ss << R"""(
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
)""";
    }

    if (set_called) {
      ss << R"""(
static inline void set(float* mem, float val, int64_t length) {
  for (int64_t i = 0; i < length; ++i) {
    mem[i] = val;
  }
}
)""";
    }

    ss << "\n";
    const auto &sizes = memory_sizes();
    auto i = 0;
    auto num_inputs = lt.ir.inputs().size();
    auto num_outputs = lt.ir.outputs().size();
    auto dump = [&](int idx, int64_t s, bool force_nonnull = false) {
      ss << idx << ":";
      if (s <= 1 && !force_nonnull) {
        ss << "nullptr";
      } else {
        ss << "float[" << s << "]";
      }
      ss << ", ";
    };
    ss << "// memory: {\n";
    ss << "//   ";
    for (; i < num_inputs; ++i) {
      dump(i, sizes.at(i), true);
    }
    ss << "// inputs\n";
    ss << "//   ";
    for (; i < num_inputs + num_outputs; ++i) {
      dump(i, sizes.at(i), true);
    }
    ss << "// outputs\n";
    ss << "//   ";
    for (; i < sizes.size(); ++i) {
      dump(i, sizes.at(i));
    }
    ss << "// scratch\n";
    ss << "// }\n";
    ss << "void fn_" << count << "(void** memory) {\n";
    ss << body.str();
    ss << "}\n";
    return ss.str();
  }
  if (lt.kind(ref) == LoopTree::NODE) {
    return gen_node_string(ref);
  }
  return gen_loop_string(ref, overrides);
}

std::vector<void *> Compiler::allocate() const {
  auto sizes = memory_sizes();
  std::vector<void *> memory(allocations.size());
  for (auto i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > 0) {
      memory[i] = calloc(sizes[i], sizeof(float));
    }
  }
  return memory;
}

std::vector<int64_t> Compiler::memory_sizes() const {
  std::vector<int64_t> memory(allocations.size());
  for (const auto &p : allocations) {
    const auto &alloc = p.second;
    // don't allocate inputs and outputs
    if (alloc.mem_idx < lt.ir.inputs().size() + lt.ir.outputs().size()) {
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

Compiler::IdxInformation Compiler::gen_idx_info(
    LoopTree::TreeRef ref, const Compiler::Access &access) const {
  Compiler::IdxInformation info;

  ASSERT(lt.kind(ref) == LoopTree::NODE);
  ref = lt.parent(ref);
  std::unordered_map<IR::VarRef, int> var_to_max_idx;
  std::unordered_map<IR::VarRef, int64_t> last;

  while (ref != -1) {
    auto loop = lt.loop(ref);
    auto stride = ([&]() -> int64_t {
      if (last.count(loop.var)) {
        return last.at(loop.var);
      }
      if (access.vars.count(loop.var)) {
        auto t = access.vars.at(loop.var);
        auto stride = std::get<0>(t);
        auto offset = std::get<1>(t);
        auto max = std::get<2>(t);
        if (max != -1 || offset < 0) {
          if (max == -1) {
            max = var_sizes.at(loop.var);
          }
          var_to_max_idx[loop.var] = info.maxes.size();
          info.maxes.emplace_back((max - offset) * stride);
          info.mins.emplace_back(-offset * stride);
        }
        return stride;
      } else {
        return 0L;
      }
    })();
    info.offset = access.total_offset;
    info.strides.emplace(info.strides.begin(), stride);
    if (var_to_max_idx.count(loop.var)) {
      info.idxs.emplace(info.idxs.begin(), var_to_max_idx.at(loop.var));
    } else {
      info.idxs.emplace(info.idxs.begin(), -1);
    }
    if (stride) {
      last[loop.var] = stride * loop.size + loop.tail;
    }
    ref = lt.parent(ref);
  }

  return info;
}

IdxFn Compiler::gen_idx_fn(LoopTree::TreeRef ref,
                           const Compiler::Access &access) const {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  auto n = lt.node(ref);
  auto info = gen_idx_info(ref, access);

  ref = lt.parent(ref);
  if (ref == -1) {
    return [](int indices[MAX_DEPTH]) { return 0; };
  }

  if (info.maxes.size()) {
    return [info](int indices[MAX_DEPTH]) -> int64_t {
      std::vector<int64_t> totals(info.maxes.size());
      int64_t idx = 0;
      for (auto i = 0; i < info.strides.size(); ++i) {
        auto bound_idx = info.idxs[i];
        if (bound_idx != -1) {
          totals[bound_idx] += indices[i] * info.strides[i];
          if (totals[bound_idx] >= info.maxes[bound_idx]) {
            return -1L;
          }
          if (totals[bound_idx] < info.mins[bound_idx]) {
            return -1L;
          }
        }
        idx += indices[i] * info.strides[i];
      }
      return idx + info.offset;
    };
  }

  return [info, n](int indices[MAX_DEPTH]) -> int64_t {
    int64_t idx = 0;
    for (auto i = 0; i < info.strides.size(); ++i) {
      idx += indices[i] * info.strides[i];
    }
    return idx + info.offset;
  };
}

InnerFnTypeImproved Compiler::gen_mem_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  if (node.op() == Operation::read) {
    return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {};
  }

  ASSERT(node.inputs().size() == 1)
      << "Cannot call gen_mem_node on this node " << lt.ir.dump(node_ref);
  auto inacc = gen_access(node.inputs().at(0), ref);
  auto inidx = gen_idx_fn(ref, inacc);

  auto outacc = gen_access(node_ref, ref);

  auto outidx = gen_idx_fn(ref, outacc);

  auto s = lt.ir.dump(node_ref);
  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto outi = outidx(indices);
    auto ini = inidx(indices);
    if (outi >= 0 && ini >= 0) {
      ASSERT(outi < outacc.alloc.size())
          << "accessing " << outi << " out of bounds (" << outacc.alloc.size()
          << ")";
      ((float *)memory[outacc.alloc.mem_idx])[outi] =
          ((float *)memory[inacc.alloc.mem_idx])[ini];
    } else if (outi >= 0) {
      ASSERT(outi < outacc.alloc.size())
          << "accessing " << outi << " out of bounds (" << outacc.alloc.size()
          << ")";
      ((float *)memory[outacc.alloc.mem_idx])[outi] = 0;
    }
  };
}

InnerFnTypeImproved Compiler::gen_binary_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  std::vector<std::pair<int, IdxFn>> inputs;
  for (const auto &inp : node.inputs()) {
    auto inacc = gen_access(inp, ref);
    auto inidx = gen_idx_fn(ref, inacc);
    inputs.emplace_back(inacc.alloc.mem_idx, inidx);
  }

  auto outacc = gen_access(node_ref, ref);
  auto outidx = gen_idx_fn(ref, outacc);

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
    auto outi = outidx(indices);
    auto first_inp = inputs.at(0);
    auto out_f = ((float *)memory[first_inp.first])[first_inp.second(indices)];
    for (auto i = 1; i < inputs.size(); ++i) {
      auto other_inp = inputs.at(1);
      auto f = ((float *)memory[other_inp.first])[other_inp.second(indices)];
      out_f = arith(out_f, f);
    }
    if (is_reduction) {
      auto f = ((float *)memory[outacc.alloc.mem_idx])[outi];
      out_f = arith(out_f, f);
    }
    ((float *)memory[outacc.alloc.mem_idx])[outi] = out_f;
  };
}

InnerFnTypeImproved Compiler::gen_unary_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  ASSERT(node.inputs().size() == 1);
  const auto &inp = node.inputs().at(0);

  auto inacc = gen_access(inp, ref);
  auto inidx = gen_idx_fn(ref, inacc);

  auto outacc = gen_access(node_ref, ref);
  auto outidx = gen_idx_fn(ref, outacc);

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
    auto outi = outidx(indices);
    auto ini = inidx(indices);
    auto f = ((float *)memory[inacc.alloc.mem_idx])[ini];
    ((float *)memory[outacc.alloc.mem_idx])[outi] = arith(f);
  };
}

InnerFnTypeImproved Compiler::gen_add_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  std::vector<std::pair<int, IdxFn>> inputs;
  for (const auto &inp : node.inputs()) {
    auto inacc = gen_access(inp, ref);
    auto inidx = gen_idx_fn(ref, inacc);
    inputs.emplace_back(inacc.alloc.mem_idx, inidx);
  }

  auto outacc = gen_access(node_ref, ref);
  auto outidx = gen_idx_fn(ref, outacc);

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto outi = outidx(indices);
    for (const auto &p : inputs) {
      auto ini = p.second(indices);
      ((float *)memory[outacc.alloc.mem_idx])[outi] +=
          ((float *)memory[p.first])[ini];
    }
  };
}

InnerFnTypeImproved Compiler::gen_mul_node(LoopTree::TreeRef ref) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  std::vector<std::pair<int, IdxFn>> inputs;
  for (const auto &inp : node.inputs()) {
    auto inacc = gen_access(inp, ref);
    auto inidx = gen_idx_fn(ref, inacc);
    inputs.emplace_back(inacc.alloc.mem_idx, inidx);
  }

  auto outacc = gen_access(node_ref, ref);
  auto outidx = gen_idx_fn(ref, outacc);

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto outi = outidx(indices);
    ((float *)memory[outacc.alloc.mem_idx])[outi] = 1;
    for (const auto &p : inputs) {
      auto ini = p.second(indices);
      ((float *)memory[outacc.alloc.mem_idx])[outi] *=
          ((float *)memory[p.first])[ini];
    }
  };
}

InnerFnTypeImproved Compiler::gen_node(LoopTree::TreeRef ref) const {
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

struct CPUCompiled : public Compiled {
  std::vector<int64_t> intermediates;
  InnerFnTypeImproved fn;
  mutable std::vector<void *> mem;
  mutable std::vector<int64_t> mem_sizes;
  std::shared_ptr<loop_tool::DynamicLibrary> dll;

  CPUCompiled(const LoopTree &lt,
              const std::unordered_set<LoopTree::TreeRef> &threaded,
              LoopTree::TreeRef ref, const GenFnType &callback) {
    auto compiler = Compiler(lt);
    auto code = compiler.gen_string();
    try {
      std::stringstream fn_name;
      fn_name << "fn_" << compiler.count;
      std::string source_name = "/tmp/" + fn_name.str() + ".c";
      std::string lib_name = "/tmp/" + fn_name.str() + ".so";
      std::ofstream(source_name, std::ios::trunc) << code;
      std::string compile_call =
          "cc -Wall -Wno-unused-function -Wno-unused-variable -Werror -O3 "
          "-std=c99 -fpic -shared -o " +
          lib_name + " " + source_name;
      auto start = std::chrono::steady_clock::now();
      ASSERT(!std::system(compile_call.c_str()));
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = end - start;
      dll = std::make_shared<loop_tool::DynamicLibrary>(lib_name.c_str());
      auto fn_impl = dll->sym<void (*)(void **)>(fn_name.str().c_str());
      fn = [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
        fn_impl(const_cast<void **>(memory.data()));
      };
      std::remove(source_name.c_str());
    } catch (const std::exception &e) {
      std::cerr << e.what() << "\n";
      std::cerr << "Falling back to interpreted execution on CPU\n";
      fn = compiler.gen_exec();
    }

    mem = compiler.allocate();
    mem_sizes = compiler.memory_sizes();
  }

  ~CPUCompiled() {
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
};

std::unique_ptr<Compiled> CPUBackend::compile_impl(
    const LoopTree &lt, const std::unordered_set<LoopTree::TreeRef> &parallel,
    LoopTree::TreeRef root) {
  return std::make_unique<CPUCompiled>(lt, parallel, root, callback);
}

int CPUBackend::hardware_requirement() const {
  // CPU is the only guaranteed hardware, always id = 0
  return 1 << 0;
}

static RegisterBackend cpu_backend_reg_(std::make_shared<CPUBackend>());

}  // namespace loop_tool
