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

InnerFnType gen_fn(const LoopTree &lt, const Auxiliary &aux,
                   LoopTree::TreeRef ref, const GenFnType &callback);

// 0 -> not CPU
// 1 -> CPU
// 2 -> parallel CPU
int cpu_backend(const LoopTree &lt, LoopTree::TreeRef ref) {
  auto annot = lt.annotation(ref);
  if (annot == "cpu_parallel") {
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

  size_t total = 1;
  for (auto &p : var_sizes) {
    total *= (p.second.first + p.second.second);
  }

  // now we traverse upward and count the number of instances of this memory
  size_t thread_multiplier = 1;
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

std::vector<std::pair<int, size_t>> gen_idx_vector(const LoopTree &lt,
                                                   const Auxiliary &aux,
                                                   const Allocation &alloc,
                                                   LoopTree::TreeRef use) {
  std::vector<std::pair<int, size_t>> idx_vec;
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
      IR::VarRef orig_var = -1;
      for (auto v : lt.ir.vars()) {
        if (c.first.symbol().name() == lt.ir.var(v).name()) {
          orig_var = v;
        }
      }
      ASSERT(orig_var != -1)
          << "cannot find var for symbolic constraint on " << c.first.dump();
      for (auto sym : view_symbols) {
        for (auto v : user_vars) {
          if (sym.name() == lt.ir.var(v).name()) {
            user_view_vars.insert(std::make_pair(
                v, std::make_pair(differentiate(c.second, sym), orig_var)));
            mapped_view_vars[orig_var].emplace_back(v);
          }
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
      size_t size = 1;
      for (auto iv : producer_vars) {
        if (iv == v) {
          break;
        }
        if (!producer_var_loops.count(iv)) {
          continue;
        }
        size_t var_size = 1;
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
std::function<size_t(int[MAX_DEPTH])> gen_idx_func(const LoopTree &lt,
                                                   const Auxiliary &aux,
                                                   const Allocation &alloc,
                                                   LoopTree::TreeRef use) {
  auto ref = alloc.producer;
  ASSERT(lt.kind(ref) == LoopTree::NODE);
  ASSERT(lt.kind(use) == LoopTree::NODE);

  auto idx_vec = gen_idx_vector(lt, aux, alloc, use);
  return [=](int indices[MAX_DEPTH]) {
    size_t idx = 0;
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

  std::vector<std::pair<std::function<size_t(int[MAX_DEPTH])>, int>> inputs;
  std::pair<std::function<size_t(int[MAX_DEPTH])>, int> output;

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

  std::vector<std::pair<std::function<size_t(int[MAX_DEPTH])>, int>> inputs;
  std::pair<std::function<size_t(int[MAX_DEPTH])>, int> output;

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

  ASSERT(size > 0);
  ASSERT(tail_size >= 0);
  std::vector<InnerFnType> fns;
  for (auto c : tree_node.children) {
    fns.emplace_back(gen_fn(lt, aux, c, callback));
  }

  auto inner_size = aux.inner_size.at(ref);
  auto memory_fn = gen_mem(lt, aux, ref);

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
    std::unordered_map<LoopTree::TreeRef, size_t> &inner_size,
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
      ASSERT(inner_size[loop_ref] == old_size);
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
std::pair<std::function<void(const std::vector<void *> &)>, std::vector<size_t>>
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
    memory_fn(memory);
    for (const auto &fn : fns) {
      int indices[MAX_DEPTH] = {0};
      int tails[MAX_DEPTH] = {0};
      fn(memory, indices, tails);
    }
  };

  std::vector<size_t> sizes;
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
  std::vector<size_t> intermediates;
  std::function<void(const std::vector<void *> &)> fn;

  CPUCompiled(const LoopTree &lt,
              const std::unordered_set<LoopTree::TreeRef> &threaded,
              LoopTree::TreeRef ref, const GenFnType &callback) {
    std::tie(fn, intermediates) = compile(lt, callback);
  }

  void run(const std::vector<void *> &memory, bool sync) const override {
    auto memory_w_intermediates = memory;
    std::vector<void *> free_me;
    for (auto s : intermediates) {
      memory_w_intermediates.emplace_back(calloc(1, s));
      free_me.emplace_back(memory_w_intermediates.back());
    }
    fn(memory_w_intermediates);

    for (auto v : free_me) {
      free(v);
    }
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
