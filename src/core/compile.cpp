/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/compile.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <unordered_set>

#include "loop_tool/backend.h"
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
  auto producer_vars = producer.vars();  // lt.ir.loop_vars(producer_ref);
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

struct CPUCompiled : public Compiled {
  std::vector<int64_t> intermediates;
  //std::function<void(const std::vector<void *> &)> fn;
  InnerFnTypeImproved fn;
  mutable std::vector<void*> mem;

  CPUCompiled(const LoopTree &lt,
              const std::unordered_set<LoopTree::TreeRef> &threaded,
              LoopTree::TreeRef ref, const GenFnType &callback) {
    //std::cerr << "CALLING COMPILER " << lt.dump() << "\n";
    //std::cerr << "dot " << dot(lt.ir) << "\n";
    auto compiler = Compiler(lt);
    fn = compiler.gen();
    mem = compiler.allocate();

    //std::cerr << "mem size " << mem.size() << "\n";
    //ASSERT(0);
    //std::tie(fn, intermediates) = compile(lt, callback);
  }

  void run(const std::vector<void *> &memory, bool sync) const override {
    int indices[MAX_DEPTH] = {0};
    for (auto i = 0; i < memory.size(); ++i) {
      mem[i] = memory[i];
    }
    fn(mem, indices);
    for (auto i = 0; i < mem.size(); ++i) {
      std::cerr << "mem " << i << ": " << ((float*)mem[i])[0] << "\n";
      //for (auto i = 0; i < 5; ++i) {
      //  std::cerr << "idx: " << i << ": " << ((float*)m)[i] << '\n';
      //}
      //std::cerr << '\n';
    }
    //auto memory_w_intermediates = memory;
    //std::vector<void *> free_me;
    //for (auto s : intermediates) {
    //  memory_w_intermediates.emplace_back(calloc(1, s));
    //  free_me.emplace_back(memory_w_intermediates.back());
    //}
    //fn(memory_w_intermediates);

    //for (auto v : free_me) {
    //  free(v);
    //}
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

/* REWRITE */

Compiler::Compiler(const LoopTree &lt_) : lt(lt_) {
  std::vector<LoopTree::TreeRef> reverse_order;
  std::cerr << "DUMP DUMP DUMP \n" << lt.dump() << " \n======\n\n";
  lt.walk([&](LoopTree::TreeRef ref, int) { reverse_order.emplace_back(ref); });
  std::reverse(reverse_order.begin(), reverse_order.end());

  std::unordered_map<IR::VarRef, int64_t> cur_sizes;
  std::unordered_set<LoopTree::TreeRef> traversed;

  auto resolve_view = [&](IR::NodeRef n) {
    const auto &node = lt.ir.node(n);
    if (node.outputs().size() == 1) {
      const auto& consumer_ref = node.outputs().at(0);
      const auto& consumer = lt.ir.node(consumer_ref);
      if (consumer.op() == Operation::write && consumer.inputs().size() == 1) {
        return consumer_ref;
      }
    }
    if (node.op() != Operation::view) {
      return n;
    }
    while (!lt.scheduled.count(n)) {
      const auto &node = lt.ir.node(n);
      if (node.op() == Operation::read) {
        return n;
      }
      ASSERT(node.op() == Operation::view);
      ASSERT(node.inputs().size() == 1);
      n = node.inputs().at(0);
    }
    return n;
  };

  for (const auto &node_ref : lt.ir.nodes()) {
    resolved_views[node_ref] = resolve_view(node_ref);
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

  for (const auto &p : var_sizes) {
    std::cerr << lt.ir.var(p.first).name() << ": " << p.second << "\n";
  }

  // std::cerr << lt.dump([&](const LoopTree::TreeRef& t) {
  //  if (inner_sizes.count(t)) {
  //    return std::to_string(inner_sizes.at(t));
  //  }
  //  return std::string("");
  //}) << "\n";
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
    body_children.emplace_back(gen(cref, overrides));
  }
  if (tail > 0) {
    // find first loop of same var, and override
    overrides[loop.var] = tail;
    for (const auto &cref : lt.children(ref)) {
      tail_children.emplace_back(gen(cref, overrides));
    }
  }

  for (const auto& p : allocations) {
    std::cerr << "ALLOC ALLOC\n";
    const auto& alloc = p.second;
    if (alloc.lca == ref) {
      std::cerr << "RESET AT THIS LOO!\n";
    }
  }

  auto idx = lt.depth(ref);
  std::cerr << "IDX IS " << idx << "\n";
  auto tail_fn = [=](const std::vector<void *> &memory,
                     int indices[MAX_DEPTH]) {
    indices[idx] = size;
    for (const auto &c : tail_children) {
      c(memory, indices);
    }
  };

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    for (auto i = 0; i < size; ++i) {
      indices[idx] = i;
      //std::cerr << "SETTING IDX TO BE " << i << "\n";
      std::cerr << "CLEAR HERE\n";
      for (const auto &c : body_children) {
        c(memory, indices);
      }
    }
    tail_fn(memory, indices);
  };
}

InnerFnTypeImproved Compiler::gen(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  if (ref == -1) {
    std::vector<InnerFnTypeImproved> roots;
    for (const auto &cref : lt.roots) {
      roots.emplace_back(gen(cref, overrides));
    }
    return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
      for (const auto &fn : roots) {
        fn(memory, indices);
      }
    };
  }
  if (lt.kind(ref) == LoopTree::NODE) {
    return gen_node(ref, overrides);
  } else {
    ASSERT(lt.kind(ref) == LoopTree::LOOP);
    return gen_loop(ref, overrides);
  }
}

std::vector<void *> Compiler::allocate() const {
  std::vector<void *> memory(allocations.size());
  for (const auto &p : allocations) {
    const auto &alloc = p.second;
    // don't allocate inputs and outputs
    if (alloc.mem_idx < lt.ir.inputs().size() + lt.ir.outputs().size()) {
      continue;
    }
    size_t size = 1;
    for (auto s : alloc.sizes) {
      size *= s > 0 ? s : 1;
    }
    memory[alloc.mem_idx] = calloc(size, sizeof(float));
  }
  return memory;
}

/*
  stride(loop_var) = stride(base_var) * diff(eq, loop_var)
  offset(node) = eq @ all zero

  z -> i + j
  p <- i + 1
  p -> x + k

  x <- (p - k)
  x <- ((i + 1) - k)
  i <- z - j
  x <- z - j + 1 - k
  off(x) = 1
  base var is z, isolate it
  z <- x + j - 1 + k
  stride(x) = diff(z, x) * stride(z) = 1
  off(x) = off(z) = -1


  [ x x x ]
  y = x + 1
  [ 0 x x x 0 ]
  y = j
  j = x + 1
  j - 1 = x
  offset(x) = -1
  j + -1



  given a read (ref to node) and a loop nest, we
  determine exactly the input expression relative to the loop
  if the output

*/

std::vector<symbolic::Constraint> Compiler::gen_constraints(
    IR::NodeRef node_ref, LoopTree::TreeRef ref) const {
  // Find a route to a scheduled base node
  auto base_node_ref = resolved_views.at(node_ref);
  std::cerr << "   node "<<lt.ir.dump(node_ref)<<" ->base node " << lt.ir.dump(base_node_ref) << "\n";
  const auto &node = lt.ir.node(lt.node(ref));
  if (node.op() == Operation::view) {
    ASSERT(node.inputs().size() == 1);
    base_node_ref = resolved_views.at(node.inputs().at(0));
  }

  std::vector<Constraint> constraints;

  // collect initial constraints
  auto vars = to_set(node.vars());
  for (auto c : node.constraints()) {
    std::cerr << "   lookin at " << c.first.dump() <<"\n";
    if (c.first.type() != Expr::Type::symbol) {
      std::cerr << "  skippin \n";
      continue;
    }
    auto sym = c.first.symbol();
    if (node.has_sym(sym) && vars.count(node.var(sym))) {
      constraints.emplace_back(c);
    }
  }

  // eager exit if we don't have to calculate anything
  if (node_ref == base_node_ref) {
    return constraints;
  }

  // begin to coalesce constraints
  auto cur_node_ref = node.inputs().at(0);

  while (cur_node_ref != base_node_ref) {
    auto &cur_node = lt.ir.node(cur_node_ref);
    for (auto c : cur_node.constraints()) {
      if (c.first.type() != Expr::Type::symbol) {
        continue;
      }
      auto sym = c.first.symbol();
      bool valid = true;
      for (auto cc : constraints) {
        if (c.second.contains(cc.first.symbol())) {
          valid = false;
        }
      }
      if (valid) {
        for (auto &cc : constraints) {
          if (cc.second.contains(sym)) {
            cc.second = cc.second.replace(sym, c.second);
          }
        }
      }
    }

    if (!cur_node.inputs().size()) {
      break;
    }
    cur_node_ref = cur_node.inputs().at(0);
  }

  // On occassion (e.g. windowed constraints), we pick up
  // dependencies on output variables.  We can safely set these to zero
  // for calculation of offsets and derivatives
  // (They'd go to zero anyway for the calculations)
  for (auto &cc : constraints) {
    for (auto sym : cc.second.symbols()) {
      if (node.has_sym(sym) && vars.count(node.var(sym))) {
        cc.second = cc.second.replace(sym, Expr(0)).simplify();
      }
    }
  }
  return constraints;
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

  // auto alloc = allocations.at(node_ref);
  const auto &node = lt.ir.node(node_ref);
  if (!lt.scheduled.count(node_ref)) {
    std::vector<int64_t> sizes;
    if (node.op() == Operation::read || node.op() == Operation::write) {
      std::cerr << "SIZE FO READ "<<lt.ir.dump(node_ref) <<" (num vars "<<node.vars().size()<<")";
      for (auto v : node.vars()) {
        // std::cerr << lt.ir.dump(node_ref) << "\n";
        std::cerr << var_sizes.at(v) << " ";
        sizes.emplace_back(var_sizes.at(v));
      }
      std::cerr << "\n";
      return Allocation(mem_idx, sizes, -1, -1);
    }
    return Allocation(mem_idx);
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
    // std::cout << "LCA IS " << lca << " ADN REF IS " << tr << "\n";
    lca = lt.lca(lca, tr);
  }
  if (node.op() == Operation::write || node.op() == Operation::read) {
    lca = -1;
  }

  std::unordered_map<IR::VarRef, int64_t> var_sizes;
  while (ref != lca) {
    auto loop = lt.loop(ref);
    // std::cerr << lt.ir.var(loop.var).name() << " " << loop.size << "r" <<
    // loop.tail << "\n";
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
      sizes.emplace_back(var_sizes.at(v));
    } else {
      sizes.emplace_back(0);
    }
  }
  return Allocation(mem_idx, sizes, lca, -1);
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
   if "real" is x' and collected is x, then offset is -1 (negatives are always
 skipped)
*/
Compiler::Access Compiler::gen_access(IR::NodeRef node_ref,
                                      LoopTree::TreeRef ref) const {
  /*
    find all the expressions for the output variables.
    follow inputs and substitute their expressions
    return final expressions
  */
  std::cerr << "\n";
  auto view_exprs = gen_constraints(node_ref, ref);
  std::cerr << "==== ANALYSING === " << lt.ir.dump(node_ref) << " (num constraints: " << view_exprs.size() << ")\n";

  auto base_node_ref = resolved_views.at(node_ref);
  const auto &base_node = lt.ir.node(base_node_ref);
  auto alloc = allocations.at(base_node_ref);
  std::cerr << "alloc for " << lt.ir.dump(base_node_ref) << ": ";
  for (auto s : alloc.sizes) {
    std::cerr << s << ' ';
  }
  std::cerr << " @ idx " << alloc.mem_idx;
  std::cerr << "\n";

  auto use_node_ref = lt.node(ref);
  const auto &use_node = lt.ir.node(use_node_ref);

  auto node_vars = to_set(lt.ir.all_vars(use_node_ref));
  auto scope_vars = lt.scope_vars(ref);
  std::cerr << "scope vars size " << scope_vars.size() << ",";
  std::cerr << " vars: ";
  for (auto v : scope_vars) {
  std::cerr << lt.ir.var(v).name() << " ";
  }
  std::cerr << "\n";
  std::cerr << "number of available constraints: " << view_exprs.size() << "\n";
  auto vars = intersection(node_vars, scope_vars);
  bool is_write = use_node_ref == node_ref;
  std::cerr << "we're " << (is_write ? "writing to" : "reading from")
            << " base node %" << base_node_ref << " with vars ";

  // either output or input vars
  std::unordered_set<symbolic::Symbol, Hash<symbolic::Symbol>> base_symbols;
  std::cerr << "(base vars ";
  for (auto v : base_node.vars()) {
  if (var_to_sym.count(v)) {
    base_symbols.insert(var_to_sym.at(v));
    }
    std::cerr << lt.ir.var(v).name() << " ";
  }
  std::cerr << ") ";


  auto has_base_symbol = [&](const symbolic::Constraint& c) {
    for (auto s : c.first.symbols()) {
      if (base_symbols.count(s)) {
      return true;
      }}
    for (auto s : c.second.symbols()) {
      if (base_symbols.count(s)) {
      return true; }}
      return false;
  };

  auto get_base_symbol = [&](const symbolic::Constraint &c) {
    symbolic::Symbol base_sym;
    bool found_base = false;
    for (auto s : c.first.symbols()) {
      if (base_symbols.count(s)) {
        ASSERT(!found_base) << "Found multiple base symbols in constraint "
                            << c.first.dump() << ": " << c.second.dump();
        base_sym = s;
        found_base = true;
      }
    }
    for (auto s : c.second.symbols()) {
      if (base_symbols.count(s)) {
        ASSERT(!found_base) << "Found multiple base symbols in constraint "
                            << c.first.dump() << ": " << c.second.dump();
        base_sym = s;
        found_base = true;
      }
    }
    std::cerr << "symbols ";
    for (auto s : base_symbols) {
      std::cerr << symbolic::Expr(s).dump() << " ";
    }
    std::cerr << "\n";
    ASSERT(found_base) << "Couldn't find base symbol in constraint "
                       << c.first.dump() << ": " << c.second.dump();
    return base_sym;
  };

  auto zero = [&](const symbolic::Expr &expr) {
    auto sized = expr.walk([&](const symbolic::Expr &e) {
      if (e.op() == symbolic::Op::size) {
        auto arg = e.args().at(0);
        if (arg.type() == Expr::Type::symbol) {
          auto s = var_sizes.at(sym_to_var.at(arg.symbol()));
          return Expr(s);
        }
      }
      return e;
    }).simplify();
    return sized.walk([&](const symbolic::Expr &e) {
      if (e.type() == Expr::Type::symbol) {
        return Expr(0);
      }
      return e;
    }).simplify();
    auto out = expr;
    for (auto s : expr.symbols()) {
      out = out.replace(s, 0).simplify();
    }
    return out;
  };

  auto stride_at = [&](int idx) {
    int64_t stride = alloc.sizes.at(idx) > 0 ? 1 : 0;
    for (auto i = idx + 1; i < alloc.sizes.size(); ++i) {
      auto size = alloc.sizes.at(i);
      stride *= size > 0 ? size : 1;
    }
    return stride;
  };
  std::unordered_map<IR::VarRef, int64_t> base_strides;
  for (auto i = 0; i < base_node.vars().size(); ++i) {
    auto v = base_node.vars().at(i);
    base_strides[v] = stride_at(i);
    //std::cerr << symbolic::Expr(sym).dump() << " stride " << stride_at(i) <<"\n";
  }

  Access access(allocations.at(base_node_ref));
  for (auto v : vars) {
    std::cerr << lt.ir.var(v).name() << " ";
    // NB: ok to generate a fake symbol
    auto sym = var_to_sym.count(v) ? var_to_sym.at(v) : Symbol();
    bool set = false;
    for (auto c : view_exprs) {
      //std::cerr << "checking expr " << c.first.dump() << " = " << c.second.dump() << "\n";
      if ((c.first.contains(sym) || c.second.contains(sym)) && has_base_symbol(c)) {
        auto expr = isolate(c, sym).second;
        auto base_sym = get_base_symbol(c);
        auto base_var = sym_to_var.at(base_sym);
        auto base_expr = isolate(c, base_sym).second;
        auto stride = (base_sym == sym)
                          ? Expr(base_strides.at(v))
                          : differentiate(expr, base_sym).simplify();
        auto offset = (base_sym == sym)
                          ? Expr(0)
                          : zero(base_expr).simplify();
        //std::cerr << "offset expr " << base_expr.dump() << " " << zero(base_expr).dump() << " ";
        auto max = var_sizes.at(base_var);
        auto v_max = var_sizes.at(v);
        if (max >= v_max) {
          max = -1;
        }
        access.vars[v] = std::make_tuple(stride.value() * base_strides.at(base_var),
                                    offset.value(), max);
        //std::cerr << "(stride " << stride.dump() << " offset " << offset.dump();
        //std::cerr << " max " << max << ") ";
        //std::cerr << "(base expr " << base_expr.dump() << ") ";
        set = true;
        break;
      }
    }
    if (!set) {
      // if this var is unrelated, we don't stride
      auto stride = base_strides.count(v) ? base_strides.at(v) : 0;
      std::cerr << "striding over " << lt.ir.var(v).name() << " for node " << lt.ir.dump(base_node_ref) << " with stride " << stride << "\n";
      //std::cerr << "(stride " << stride << ")";
      access.vars[v] = std::make_tuple(stride, 0, -1);
    }
  }
  //std::cerr << "\n";
  return access;

  //std::cerr << "searching thru these exprs:\n";
  //for (auto c : view_exprs) {
  //  std::cerr << c.first.dump() << " = " << c.second.dump() << "\n";
  //}

  std::unordered_set<symbolic::Symbol, Hash<symbolic::Symbol>> symbols;
  auto pref = lt.parent(ref);
  while (pref != -1) {
    symbols.insert(var_to_sym.at(lt.loop(pref).var));
    pref = lt.parent(pref);
  }

  bool padded_write = false;  // name for when an input var is looped over
  std::unordered_map<IR::VarRef, int64_t> var_strides;
  std::unordered_map<IR::VarRef, int64_t> var_mins;
  std::unordered_map<IR::VarRef, int64_t> var_maxs;

  for (auto c : view_exprs) {
    for (auto sym : symbols) {
      ASSERT(!(c.first.contains(sym) & c.second.contains(sym)))
          << c.first.dump() << ": " << c.second.dump();
      if (c.first.contains(sym)) {
        auto var = sym_to_var.at(sym);
        auto dep_syms = c.second.symbols();
        ASSERT(dep_syms.size() == 1);
        auto dep_sym = dep_syms.at(0);
        var_strides[var] = differentiate(c.second, dep_sym).simplify().value();
        var_mins[var] = c.second.replace(dep_syms.at(0), 0).simplify().value();
        // std::cerr << sym.name() << " stride " << var_strides[var] << " offset
        // " << var_mins[var] << "\n"; var_maxs[var] =
        // c.second.replace(dep_syms.at(0), ??).simplify().value();
        // pass
      } else if (c.second.contains(sym)) {
        ASSERT(use_node.op() == Operation::view);
        padded_write = true;
        ASSERT(0) << "padded writes not yet supported";
      }
    }
  }

  for (auto v : vars) {
  }

  return Compiler::Access(Allocation());
}

IdxFn Compiler::gen_idx_fn(
    LoopTree::TreeRef ref, const Compiler::Access &access) const {
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  //std::cerr << "stride for " << lt.ir.dump(lt.node(ref)) << " ";
  ref = lt.parent(ref);
  if (ref == -1) {
    return [](int indices[MAX_DEPTH]) { return 0; };
  }
  std::vector<int64_t> strides;
  int64_t total_offset = 0;
  std::unordered_map<IR::VarRef, int> var_to_max_idx;
  std::vector<int> max_idxs;  // -1 means no max
  std::vector<int64_t> maxes;
  std::vector<int64_t> offsets;

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
          var_to_max_idx[loop.var] = maxes.size();
          maxes.emplace_back(max - offset);
          offsets.emplace_back(offset);
        }
        total_offset += offset * stride;
        //std::cerr << "STRIDE(0): " << lt.ir.var(loop.var).name() << " " << stride << "\n";
        return stride;
      } else {
        return 0L;
      }
    })();
    strides.emplace(strides.begin(), stride);
    if (var_to_max_idx.count(loop.var)) {
      max_idxs.emplace(max_idxs.begin(), var_to_max_idx.at(loop.var));
    } else {
      max_idxs.emplace(max_idxs.begin(), -1);
    }
    if (stride) {
      last[loop.var] = stride * loop.size + loop.tail;
      //std::cerr << "STRIDE: " << lt.ir.var(loop.var).name() << " " << last.at(loop.var) << "\n";
    }
    ref = lt.parent(ref);
  }
  //std::cerr << "\n";
  if (maxes.size()) {
    return [strides, total_offset, max_idxs,
            maxes, offsets](int indices[MAX_DEPTH]) -> int64_t {
      std::vector<int64_t> totals(maxes.size());
      int64_t idx = 0;
      for (auto i = 0; i < strides.size(); ++i) {
        auto max_idx = max_idxs[i];
        if (max_idx != -1) {
          totals[max_idx] += indices[i] * strides[i];
          if (totals[max_idx] >= maxes[max_idx]) {
            return -1L;
          }
          if (totals[max_idx] + offsets[max_idx] < 0) {
            return -1L;
          }
        }
        idx += indices[i] * strides[i];
      }
      return idx + total_offset;
    };
  }
  return [strides, total_offset](int indices[MAX_DEPTH]) -> int64_t {
    int64_t idx = 0;
    for (auto i = 0; i < strides.size(); ++i) {
      idx += indices[i] * strides[i];
    }
    return idx + total_offset;
  };
}

InnerFnTypeImproved Compiler::gen_mem_node(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  if (node.op() == Operation::read) {
    return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    };
  }

  ASSERT(node.inputs().size() == 1)
      << "Cannot call gen_mem_node on this node " << lt.ir.dump(node_ref);
  //auto inalloc = allocations.at(node.inputs().at(0));
  auto inacc = gen_access(node.inputs().at(0), ref);
  auto inidx = gen_idx_fn(ref, inacc);

  //auto outalloc = allocations.at(node_ref);
  auto outacc = gen_access(node_ref, ref);
  auto outidx = gen_idx_fn(ref, outacc);

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto outi = outidx(indices);
    auto ini = inidx(indices);
    //std::cerr << outalloc.mem_idx << "["<<outi<<"] <- " << inalloc.mem_idx << "["<<ini<<"]\n";
    if (outi >= 0 && ini >= 0) {
      ((float *)memory[outacc.alloc.mem_idx])[outi] =
          ((float *)memory[inacc.alloc.mem_idx])[ini];
    } else if (outi >= 0) {
      ((float *)memory[outacc.alloc.mem_idx])[outi] = 0;
    }
  };
}

InnerFnTypeImproved Compiler::gen_add_node(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  std::vector<std::pair<int, IdxFn>> inputs;
  for (const auto& inp : node.inputs()) {
    auto inacc = gen_access(inp, ref);
    auto inidx = gen_idx_fn(ref, inacc);
    inputs.emplace_back(inacc.alloc.mem_idx, inidx);
  }

  auto outacc = gen_access(node_ref, ref);
  auto outidx = gen_idx_fn(ref, outacc);

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto outi = outidx(indices);
    for (const auto& p : inputs) {
      auto ini = p.second(indices);
     // std::cerr << outacc.alloc.mem_idx << "["<<outi<<"] += " << p.first << "["<<ini<<"] {"
     //   << 
     //   ((float *)memory[p.first])[ini]
     // << "}\n";
      ((float *)memory[outacc.alloc.mem_idx])[outi] +=
        ((float *)memory[p.first])[ini];
    }
  };
}

InnerFnTypeImproved Compiler::gen_mul_node(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  std::vector<std::pair<int, IdxFn>> inputs;
  for (const auto& inp : node.inputs()) {
    auto inacc = gen_access(inp, ref);
    auto inidx = gen_idx_fn(ref, inacc);
    inputs.emplace_back(inacc.alloc.mem_idx, inidx);
  }

  auto outacc = gen_access(node_ref, ref);
  auto outidx = gen_idx_fn(ref, outacc);

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    auto outi = outidx(indices);
      ((float *)memory[outacc.alloc.mem_idx])[outi] = 1;
    for (const auto& p : inputs) {
      auto ini = p.second(indices);
      ((float *)memory[outacc.alloc.mem_idx])[outi] *=
        ((float *)memory[p.first])[ini];
    }
  };
}

InnerFnTypeImproved Compiler::gen_node(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);
  switch (node.op()) {
    case Operation::read:
    case Operation::view:
    case Operation::write:
      return gen_mem_node(ref, overrides);
    case Operation::add:
      return gen_add_node(ref, overrides);
    case Operation::multiply:
      return gen_mul_node(ref, overrides);
    default:
      ASSERT(0) << "Cannot generate node: " << lt.ir.dump(node_ref);
      return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
        ASSERT(0);
      };
  }
}

InnerFnTypeImproved Compiler::gen_node_old(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  auto node_ref = lt.node(ref);
  //std::cerr << "GEN NODE " << lt.ir.dump(node_ref) << "\n";
  const auto &node = lt.ir.node(node_ref);

  // determine if its an output or input view based on loops
  if (node.op() == Operation::view) {
    int io_val = -1;  // 0 means loop over input, 1 means loop over output vars
    std::unordered_set<symbolic::Symbol, Hash<symbolic::Symbol>> symbols;
    auto input_vars = to_set(lt.ir.node(node.inputs().at(0)).vars());
    auto output_vars = to_set(node.vars());

    auto pref = lt.parent(ref);
    while (pref != -1) {
      symbols.insert(var_to_sym.at(lt.loop(pref).var));
      pref = lt.parent(pref);
    }
    for (const auto &sym : symbols) {
      // std::cerr << "sym " << sym.name() << "\n";
      auto v = node.var(sym);
      if (input_vars.count(v)) {
        ASSERT(io_val == -1 || io_val == 0)
            << "can't loop over both input and output vars";
        io_val = 0;
      } else if (output_vars.count(v)) {
        ASSERT(io_val == -1 || io_val == 1)
            << "can't loop over both input and output vars";
        io_val = 1;
      } else {
        ASSERT(0) << "unknown var " << lt.ir.var(node.var(sym)).name();
      }
    }

    std::unordered_map<IR::VarRef, symbolic::Expr> map;
    for (const auto &v_ref : output_vars) {
      for (const auto &c : node.constraints()) {
        if (c.first.type() != Expr::Type::symbol) {
          continue;
        }
        auto sym = c.first.symbol();
        if (!node.has_sym(sym)) {
          continue;
        }
        if (node.var(sym) == v_ref) {
          map.emplace(v_ref, c.second);
        }
      }
    }
    for (const auto &v : output_vars) {
      if (!map.count(v)) {
        continue;
      }

      auto min_expr = map.at(v);
      auto max_expr = map.at(v);

      auto syms = to_set<symbolic::Symbol, symbolic::Hash>(min_expr.symbols());
      for (const auto &sym : syms) {
        min_expr = min_expr.replace(sym, Expr(0)).simplify();
      }
      for (const auto &c : node.constraints()) {
        if (c.first.type() == Expr::Type::symbol) {
          continue;
        }
        auto sym = c.first.args().at(0).symbol();
        if (syms.count(sym)) {
          max_expr = max_expr.replace(sym, c.second - Expr(1)).simplify();
        }
      }
      //std::cerr << "var " << lt.ir.var(v).name() << " has offset "
      //          << min_expr.dump() << "\n";
      //std::cerr << "var " << lt.ir.var(v).name() << " has max "
      //          << max_expr.dump() << "\n";
    }

    if (io_val == 0) {
      std::cerr << "OUTPUT VIEW, LETS CALCULATE AN OFFSET\n";
    } else if (io_val == 1) {
      std::cerr << "INPUT VIEW, LETS CALCULATE A RANGE RESTRICTION\n";
    }
  }

  for (auto input_node_ref : lt.ir.node(node_ref).inputs()) {
    std::cerr << "READ\n";
    auto access = gen_access(input_node_ref, ref);
    // auto strides = access.strides;
    // auto offset = access.offset;
    // auto ranges = access.ranges;
    // auto access_fn = [=](void* memory, int indices[MAX_DEPTH]) {
    //  int idx = offset;
    //  for (auto i = 0; i < MAX_DEPTH; ++i) {
    //    idx += indices[i] * strides[i];
    //  }
    //  for (const auto& range : ranges) {
    //    auto var_idx = offset;
    //    for (auto i = 0; i < MAX_DEPTH; ++i) {
    //      var_idx += indices[i] * range.strides[i];
    //    }
    //    if (var_idx > range.max || var_idx < range.min) {
    //      return range.fill;
    //    }
    //  }
    //  return ((float*)memory)[idx];
    //};
  }

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
    // for (auto& access_fn : inputs) {
    //  access_fn(memory, indices);
    //}
    // if (exec_fn(indices)) {
    //  memory[output][out_idx(indices)] = memory[input][in_idx(indices)];
    //}

    // for (auto i  = 0; i < MAX_DEPTH; ++i) {
    //  std::cerr << indices[i] << ", ";
    //}
    // std::cerr << "\n";
  };
}

}  // namespace loop_tool
