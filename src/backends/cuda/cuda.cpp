/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>

#include "cuda_backend.h"
#include "loop_tool/backend.h"
#include "loop_tool/compile.h"
#include "loop_tool/error.h"
#include "loop_tool/ir.h"

#define NVRTC_SAFE_CALL(x)                                                  \
  do {                                                                      \
    nvrtcResult result = x;                                                 \
    ASSERT(result == NVRTC_SUCCESS) << "\nerror: " #x " failed with error " \
                                    << nvrtcGetErrorString(result) << '\n'; \
  } while (0)

namespace loop_tool {

namespace {
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};
}  // namespace

// for manual unrolling/vectorization
using UnrollMap =
    std::unordered_map<std::pair<IR::VarRef, int>, int, pair_hash>;

std::string indent(int depth) {
  std::stringstream s;
  for (auto i = 0; i < depth + 1; ++i) {
    s << " ";
  }
  return s.str();
};

size_t count_threads(const LoopTree &lt, const CudaAux &cuda_aux,
                     LoopTree::TreeRef ref);
size_t count_parent_threads(const LoopTree &lt, const CudaAux &cuda_aux,
                            LoopTree::TreeRef ref);
size_t thread_scope(const LoopTree &lt, const CudaAux &cuda_aux,
                    LoopTree::TreeRef ref);

// full access to memory + idx + optional vector idx
std::string gen_access(const LoopTree &lt, const Auxiliary &aux,
                       const Allocation &alloc, LoopTree::TreeRef use,
                       const UnrollMap &unroll, int external_idx = -1) {
  std::stringstream ss;

  std::vector<LoopTree::Loop> parent_chain;
  auto parent = lt.parent(use);
  while (parent != -1) {
    parent_chain.emplace_back(lt.loop(parent));
    parent = lt.parent(parent);
  }
  std::reverse(parent_chain.begin(), parent_chain.end());
  auto order = parent_chain;  // lt.loop_order(lt.node(use));
  auto idx_vec = gen_idx_vector(lt, aux, alloc, use);
  // can we map innermost dims to vector index?
  // this becomes false if non-innermost sizes cannot be vectorized (TODO relax)
  bool unrolled_vectorize = true;
  for (const auto &p : idx_vec) {
    unrolled_vectorize &= (p.second % 4 == 0);
    auto loop = order[p.first];
    std::pair<IR::VarRef, int> key = {loop.var, loop.var_depth};
    if (!unroll.count(key)) {
      break;
    }
  }

  bool vectorize = false;
  bool flatten = true;  // cast from float4 to float
  if (alloc.size % 4 == 0) {
    // we can index directly into the vector
    if (unrolled_vectorize) {
      vectorize = true;
      flatten = false;
    } else {
      flatten = true;
    }
  }

  // memory name
  if (flatten) {
    ss << "((float*)";
  }
  if (external_idx > -1) {  // for reads/writes to real memory
    ss << "ext_" << external_idx;
  } else {
    ss << "mem_" << alloc.idx;
  }
  if (flatten) {
    ss << ")";
  }

  // memory index
  bool innermost = true;
  std::string extra = "";
  ss << "[";
  for (const auto &p : idx_vec) {
    auto loop = order.at(p.first);
    auto size = p.second;
    if (!innermost && vectorize) {
      ASSERT(size % 4 == 0);
    }
    std::pair<IR::VarRef, int> key = {loop.var, loop.var_depth};
    if (unroll.count(key)) {
      ss << unroll.at(key) * size / (vectorize ? 4 : 1);
      if (innermost && vectorize) {
        extra = "." + (std::vector<std::string>(
                          {"x", "y", "z", "w"})[unroll.at(key) % 4]);
      }
    } else {
      auto s = (size / (vectorize ? 4 : 1));
      ss << lt.ir.var(loop.var).name() << "_" << loop.var_depth << " * " << s;
    }
    ss << " + ";
    if (innermost) {
      innermost = false;
    }
  }
  ss << "0";  // keeps the expression well formed
  ss << "]" << extra;

  return ss.str();
};

std::string gen_compute(const LoopTree &lt, const Auxiliary &aux,
                        UnrollMap &unroll, LoopTree::TreeRef ref,
                        std::string sym) {
  std::stringstream ss;
  auto node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);
  bool reduction = (lt.ir.pointwise_vars(node_ref).size() !=
                    lt.ir.all_vars(node_ref).size());
  ss << gen_access(lt, aux, aux.allocs.at(node_ref), ref, unroll) << " ";
  if (reduction) {
    ss << sym;
  }
  ss << "= ";
  for (const auto &inp : node.inputs()) {
    auto idx = aux.allocs.at(node_ref).idx;
    auto inp_alloc = aux.allocs.at(inp);
    auto inp_idx = inp_alloc.idx;
    ss << gen_access(lt, aux, aux.allocs.at(inp), ref, unroll);
    if (&inp != &node.inputs().back()) {
      ss << " " << sym << " ";
    }
  }
  ss << ";\n";
  return ss.str();
}

std::string gen_node(const LoopTree &lt, const Auxiliary &aux,
                     UnrollMap &unroll, LoopTree::TreeRef ref) {
  std::stringstream ss;
  auto depth = lt.tree_node(ref).depth;
  auto node_ref = lt.node(ref);
  auto out_alloc = aux.allocs.at(node_ref);
  const auto &node = lt.ir.node(node_ref);
  if (node.op() == "add") {
    ss << indent(depth);
    ss << gen_compute(lt, aux, unroll, ref, "+");
  } else if (node.op() == "mul") {
    ss << indent(depth);
    ss << gen_compute(lt, aux, unroll, ref, "*");
  } else if (node.op() == "read") {
    int external_memory = -1;
    for (auto i = 0; i < lt.ir.inputs().size(); ++i) {
      if (lt.ir.inputs()[i] == lt.node(ref)) {
        external_memory = i;
      }
    }
    ASSERT(external_memory > -1 && "No input found!");
    auto out_alloc = aux.allocs.at(node_ref);
    auto inp_alloc = out_alloc;
    inp_alloc.lca = -1;  // TODO clean up read hacks

    ss << indent(depth);
    ss << gen_access(lt, aux, out_alloc, ref, unroll);
    ss << " = ";
    ss << gen_access(lt, aux, inp_alloc, ref, unroll, external_memory);
    ss << ";\n";
  } else if (node.op() == "write") {
    int external_memory = -1;
    for (auto i = 0; i < lt.ir.outputs().size(); ++i) {
      if (lt.ir.outputs()[i] == lt.node(ref)) {
        external_memory = i + lt.ir.inputs().size();
      }
    }
    ASSERT(external_memory > -1 && "No output found!");

    auto out_alloc = aux.allocs.at(node_ref);
    ASSERT(node.inputs().size() == 1);
    auto inp_alloc = aux.allocs.at(node.inputs().at(0));

    ss << indent(depth);
    ss << gen_access(lt, aux, out_alloc, ref, unroll, external_memory);
    ss << " = ";
    ss << gen_access(lt, aux, inp_alloc, ref, unroll);
    ss << ";\n";
  }

  return ss.str();
}

std::string gen_cuda(const LoopTree &lt, const Auxiliary &aux,
                     const CudaAux &cuda_aux, UnrollMap &unroll,
                     LoopTree::TreeRef ref);

std::string gen_mem_decl(const LoopTree &lt, const Auxiliary &aux,
                         const CudaAux &cuda_aux, LoopTree::TreeRef ref,
                         bool declare = true) {
  std::stringstream ss;
  auto depth = ref > -1 ? lt.tree_node(ref).depth : 0;
  std::vector<Allocation> reset_allocs =
      aux.resets.count(ref) ? aux.resets.at(ref) : std::vector<Allocation>{};
  // we can traverse producer to LCA and check for threadedness
  // This is true because of the threading self-consistency invariant
  // If need-be this could change, but it gets really messy
  auto is_shared = [&](const Allocation &alloc) {
    auto p = lt.parent(alloc.producer);
    bool shared = false;
    while (p != alloc.lca) {
      if (cuda_aux.threaded.count(p)) {
        shared = true;
        break;
      }
      p = lt.parent(p);
    }
    if (shared && (p != -1)) {
      // ASSERT(0) << "known shared memory issue when it's not at global top
      // level scope";
    }
    return shared;
  };
  for (auto alloc : reset_allocs) {
    if (lt.ir.node(lt.node(alloc.producer)).outputs().size() == 0) {
      continue;
    }
    ss << indent(depth + 1);
    auto shared = is_shared(alloc);
    if (shared) {
      ss << "__shared__ ";
    }
    // always store to float4 if possible
    auto numel = alloc.size;
    if (alloc.size % 4 == 0) {
      ss << (declare ? "float4 " : "");
      numel = alloc.size / 4;
    } else {
      ss << (declare ? "float " : "");
    }
    ss << "mem_" << alloc.idx;
    ss << "[" << numel << "]";
    if (alloc.should_init) {
      if (shared) {
        ss << "; for (int s_i = 0; s_i < " << numel << "; ++s_i) { ";
        ss << "mem_" << alloc.idx << "[s_i] = ";
        if (alloc.size % 4 == 0) {
          ss << "make_float4(" << alloc.init_val << ", " << alloc.init_val
             << ", " << alloc.init_val << ", " << alloc.init_val << ")";
        } else {
          ss << alloc.init_val;
        }
        ss << "; };\n";
      } else {
        ss << " = {" << alloc.init_val << "};\n";
      }
    } else {
      ss << ";\n";
    }
  }
  return ss.str();
}

// basically, call this after every loop/node gen.  If there's a reduction
// this will emit after innermost node, otherwise it'll emit at the end of top
// level loop
std::string gen_sync(const LoopTree &lt, const Auxiliary &aux,
                     const CudaAux &cuda_aux, LoopTree::TreeRef ref) {
  std::stringstream ss;
  auto depth = ref > -1 ? lt.tree_node(ref).depth : 0;
  if (cuda_aux.syncs.count(ref)) {
    if (count_threads(lt, cuda_aux, ref) > cuda_aux.threads_per_block) {
      ss << "#error CANNOT COMPILE, too many threads to sync\n";
    }
    ss << indent(depth) << "__syncthreads();\n";
  }
  return ss.str();
}

std::string gen_loop(const LoopTree &lt, const Auxiliary &aux,
                     const CudaAux &cuda_aux, UnrollMap &unroll,
                     LoopTree::TreeRef ref) {
  std::stringstream ss;
  auto depth = lt.tree_node(ref).depth;
  auto loop = lt.loop(ref);
  auto v = lt.ir.var(loop.var).name();
  auto v_depth = loop.var_depth;

  // First emit the main loop, then the tail loop (if there is a tail)
  bool is_tail = cuda_aux.tail.count(loop.var) && cuda_aux.tail.at(loop.var);
  auto inner_size = aux.inner_size.at(ref);
  int loop_size =
      is_tail ? (cuda_aux.tail.at(loop.var) / inner_size) : loop.size;
  ASSERT(loop.size >= loop_size);
  int tail_size =
      is_tail ? (cuda_aux.tail.at(loop.var) % inner_size) : loop.tail;
  if (is_tail) {
    auto consumed = loop_size ? loop_size * inner_size + tail_size : 0;
    const_cast<CudaAux &>(cuda_aux).tail[loop.var] =
        cuda_aux.tail.at(loop.var) - consumed;
  }
  std::stringstream v_ss;
  v_ss << v << "_" << v_depth;
  auto v_str = v_ss.str();

  if (loop_size) {
    if (cuda_aux.unrolled.count(ref)) {
      ASSERT(loop_size > -1) << "Can only unroll sized loops";
      ASSERT(!cuda_aux.threaded.count(ref))
          << "Can only unroll non-threaded loops";
      std::pair<IR::VarRef, int> key = {loop.var, loop.var_depth};

      if (is_tail) {
        auto consumed = loop_size ? loop_size * inner_size + tail_size : 0;
        ss << indent(depth) << "// tail! " << consumed << " consumed "
           << cuda_aux.tail.at(loop.var) << " left \n";
      }
      ss << indent(depth) << "// unrolling " << v_str
         << (is_tail ? " (tail)" : "") << "\n";
      for (auto i = 0; i < loop_size; ++i) {
        auto reset_str = gen_mem_decl(lt, aux, cuda_aux, ref);
        if (reset_str.size()) {
          ss << indent(depth) << "{\n";
          ss << reset_str;
        }
        unroll[key] = i;
        for (auto c : lt.tree_node(ref).children) {
          ss << gen_cuda(lt, aux, cuda_aux, unroll, c);
        }
        if (reset_str.size()) {
          ss << indent(depth) << "}\n";
        }
      }
      unroll.erase(key);
    } else if (cuda_aux.threaded.count(ref)) {
      auto inner = cuda_aux.threaded.at(ref);
      ASSERT(inner > -1 && "Never calcualated inner size of threaded loop");
      size_t needed_threads = thread_scope(lt, cuda_aux, ref);
      // ss << indent(depth) << "if ((blockIdx.x * blockDim.x + threadIdx.x) / "
      // << 1 << " < " << needed_threads << ") ";
      ss << indent(depth) << "{\n";
      ss << indent(depth) << "int " << v_str << " = (_tid / " << inner << ") % "
         << loop.size << ";\n";
      if (loop_size != loop.size) {
        ss << indent(depth) << "if (" << v_str << " < " << loop_size << ") {\n";
      }
    } else {
      // ss << indent(depth) << "#pragma unroll\n";
      if (is_tail) {
        auto consumed = loop_size ? loop_size * inner_size + tail_size : 0;
        ss << indent(depth) << "// tail! " << consumed << " consumed "
           << cuda_aux.tail.at(loop.var) << " left \n";
      }
      ss << indent(depth) << "for (int " << v_str << " = 0;";
      ss << " " << v_str << " < " << loop_size << ";";
      ss << " ++" << v_str << ") {\n";
    }
    if (!cuda_aux.unrolled.count(ref)) {
      ss << gen_mem_decl(lt, aux, cuda_aux, ref);
      for (auto c : lt.tree_node(ref).children) {
        ss << gen_cuda(lt, aux, cuda_aux, unroll, c);
      }
      ss << indent(depth) << "}\n";
      if (cuda_aux.threaded.count(ref)) {
        if (loop_size != loop.size) {
          ss << indent(depth) << "}\n";
        }
      }
    }
  };

  if (tail_size > 0) {
    ss << indent(depth) << "// Tail logic for " << v_str << " (L" << ref
       << ")\n";
    if (cuda_aux.threaded.count(ref)) {
      auto inner = cuda_aux.threaded.at(ref);
      size_t needed_threads = thread_scope(lt, cuda_aux, ref);
      ss << indent(depth) << "if ("
         << "0 == (_tid / " << inner << ") % " << loop.size << ") {\n";
    } else {
      ss << indent(depth) << "{\n";
    }
    ss << indent(depth) << "int " << v_str << " = " << loop_size << ";\n";
    if (!is_tail) {
      ss << indent(depth) << "{ // starting tail\n";
      const_cast<CudaAux &>(cuda_aux).tail[loop.var] = tail_size;
    } else {
      ss << indent(depth) << "{ // recursive tail\n";
      const_cast<CudaAux &>(cuda_aux).tail[loop.var] = tail_size;
    }
    ss << gen_mem_decl(lt, aux, cuda_aux, ref);
    for (auto c : lt.tree_node(ref).children) {
      ss << gen_cuda(lt, aux, cuda_aux, unroll, c);
    }
    if (!is_tail) {
      ss << indent(depth) << "} // killing tail\n";
      const_cast<CudaAux &>(cuda_aux).tail[loop.var] = 0;
    } else {
      ss << indent(depth) << "} // killing recursive tail\n";
      const_cast<CudaAux &>(cuda_aux).tail[loop.var] = 0;
    }
    ss << indent(depth) << "}\n";
  }
  return ss.str();
}

// generate a guard for correct threading
std::string gen_guard(const LoopTree &lt, const Auxiliary &aux,
                      const CudaAux &cuda_aux, UnrollMap &unroll,
                      LoopTree::TreeRef ref) {
  std::stringstream ss;
  // note that any threads in sibling trees (invisible to us, but extent)
  // are *necessarily* smaller than our first threaded parent
  // thus we can find our first threaded parent i = cuda_aux.threaded.at(parent)
  // and just check `(tid % i) == 0`
  // for threaded parents that we *do not* care about (not in vars),
  // we check that `tid_var == 0`
  if (ref == -1) {
    return ss.str();
  }
  if (lt.kind(ref) == LoopTree::LOOP) {
    if (cuda_aux.threaded.count(ref)) {
      auto inner = cuda_aux.threaded.at(ref);
      auto parent = lt.parent(ref);
      auto loop = lt.loop(ref);
      auto expected_inner = loop.size * inner;
      if (cuda_aux.threaded.count(parent)) {
        auto outer = cuda_aux.threaded.at(parent);
        if (outer != expected_inner) {
          auto mod = outer / expected_inner;
          ASSERT(mod != 0)
              << "Unexpected threading mismatch cannot be reconciled";
          ss << "((_tid / " << expected_inner << ") % " << mod << " == 0)";
        }
      }
    }
    return ss.str();
  }
  auto vs = lt.ir.all_vars(lt.node(ref));
  std::unordered_set<IR::VarRef> vars(vs.begin(), vs.end());
  auto parent = lt.parent(ref);
  bool first_parent = false;
  auto last_inner = 1;
  auto last_loop_size = 1;
  while (parent != -1) {
    auto loop = lt.loop(parent);
    auto v = loop.var;
    auto v_depth = loop.var_depth;
    // we need to guard a threaded var we don't care about
    if (cuda_aux.threaded.count(parent)) {
      auto inner = cuda_aux.threaded.at(parent);
      if (vars.count(v) == 0) {
        if (ss.str().size()) {
          ss << " && ";
        }
        auto v_n = lt.ir.var(v).name();
        ss << "(" << v_n << "_" << v_depth << " == 0)";
      } else if (!first_parent) {
        first_parent = true;
        if (inner != 1) {
          if (ss.str().size()) {
            ss << " && ";
          }
          ss << "(_tid % " << inner << " == 0)";
        }
      }
    }
    parent = lt.parent(parent);
  }
  return ss.str();
}

std::string gen_cuda(const LoopTree &lt, const Auxiliary &aux,
                     const CudaAux &cuda_aux, UnrollMap &unroll,
                     LoopTree::TreeRef ref) {
  std::stringstream ss;
  ss << gen_sync(lt, aux, cuda_aux, ref);
  auto depth = lt.tree_node(ref).depth;
  auto guard = gen_guard(lt, aux, cuda_aux, unroll, ref);
  if (guard.size()) {
    ss << indent(depth) << "if (" << guard << ") {\n";
  }
  if (lt.kind(ref) == LoopTree::LOOP) {
    ss << gen_loop(lt, aux, cuda_aux, unroll, ref);
  } else {
    ss << gen_node(lt, aux, unroll, ref);
  }
  if (guard.size()) {
    ss << indent(depth) << "} // " << guard << "\n";
  }
  return ss.str();
}

std::string cuda_compile(const LoopTree &lt, const CudaAux &cuda_aux) {
  std::stringstream ss;
  auto aux = calculate_aux(lt);
  ss << "extern \"C\" __global__\nvoid kernel(";
  auto num_ext = lt.ir.inputs().size() + lt.ir.outputs().size();
  for (auto i = 0; i < num_ext; ++i) {
    ss << "float4* __restrict__ ext_" << i;
    if (i + 1 != num_ext) {
      ss << ", ";
    }
  }
  ss << ") {\n";
  ss << indent(0) << "int _tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
  UnrollMap unroll;
  ss << gen_mem_decl(lt, aux, cuda_aux, -1);
  for (auto c : lt.roots) {
    ss << gen_cuda(lt, aux, cuda_aux, unroll, c);
  }
  ss << "}\n";
  return ss.str();
}

size_t count_threads(const LoopTree &lt, const CudaAux &cuda_aux,
                     LoopTree::TreeRef ref) {
  std::vector<LoopTree::TreeRef> children;
  if (ref == -1) {
    children = lt.roots;
  } else {
    children = lt.tree_node(ref).children;
  }
  size_t max = 1;
  for (auto c : children) {
    max = std::max(count_threads(lt, cuda_aux, c), max);
  }
  if (cuda_aux.threaded.count(ref)) {
    max *= lt.loop(ref).size;
  }
  return max;
}

size_t count_parent_threads(const LoopTree &lt, const CudaAux &cuda_aux,
                            LoopTree::TreeRef ref) {
  size_t total = 1;
  if (cuda_aux.threaded.count(ref)) {
    total = cuda_aux.threaded.at(ref);  // count_threads(lt, cuda_aux, ref);
  }
  if (ref == -1) {
    return total;
  }
  auto parent = lt.parent(ref);
  while (parent != -1) {
    if (cuda_aux.threaded.count(parent)) {
      total = std::max(total, (size_t)cuda_aux.threaded.at(
                                  parent));  // lt.loop(parent).size;
    }
    parent = lt.parent(parent);
  }
  return total;
}

size_t thread_scope(const LoopTree &lt, const CudaAux &cuda_aux,
                    LoopTree::TreeRef ref) {
  std::vector<LoopTree::TreeRef> children;
  if (ref == -1) {
    children = lt.roots;
  } else {
    children = lt.tree_node(ref).children;
  }
  size_t max = 1;
  for (auto c : children) {
    max = std::max(count_threads(lt, cuda_aux, c), max);
  }
  return max;
}

void gen_cuda_kernels(const LoopTree &lt, const Auxiliary &aux,
                      const CudaAux &cuda_aux) {}

void unroll(const LoopTree &lt, CudaAux &ca) {
  const int unroll_limit = 16;  // 8 works, 16 breaks!
  lt.walk([&](LoopTree::TreeRef ref, int) {
    if (lt.kind(ref) == LoopTree::LOOP) {
      return;
    }
    auto parent = lt.parent(ref);
    auto size = 1;
    while (parent != -1) {
      size *= lt.loop(parent).size;
      if (size > unroll_limit) {
        break;
      }
      if (!ca.threaded.count(parent)) {
        ca.unrolled.insert(parent);
      }
      parent = lt.parent(parent);
    }
  });
}

constexpr int sync_global = 1;
constexpr int sync_shared = 2;
constexpr int sync_warp = 3;

void gen_threading_info(const LoopTree &lt, const Auxiliary &aux,
                        CudaAux &cuda_aux) {
  // thread scope of allocations
  std::unordered_map<IR::NodeRef, size_t> alloc_threads;
  std::unordered_map<LoopTree::TreeRef, int> syncs;
  // 1. find # threads memory is scoped to
  for (const auto &p : aux.allocs) {
    auto node_ref = p.first;
    auto num_threads = thread_scope(lt, cuda_aux, p.second.lca);
    alloc_threads[node_ref] = num_threads;
  }

  /*
  There are two types of sync points -- pointwise and reduction
  Reduction syncs happen at the node level, pointwise at the loop.
  We only consider loop-level here, node level syncs are easy to calculate
  locally.

  for a:
    for b:
      X = compute(...)
    __sync <-- just below LCA
    for c:
      for b:
        Y = compute(X, ...)
  */
  auto sync_point = [&](const Allocation &alloc, LoopTree::TreeRef ref) {
    auto parent = lt.parent(ref);
    auto trailing = ref;
    while (parent != alloc.lca) {
      trailing = parent;
      parent = lt.parent(parent);
    }
    auto from_reduce = [&]() {
      auto node_ref = lt.node(ref);
      const auto &node = lt.ir.node(node_ref);
      bool reduction = false;
      for (auto inp : node.inputs()) {
        reduction |=
            (lt.ir.pointwise_vars(inp).size() != lt.ir.all_vars(inp).size());
      }
      return reduction;
    };
    // This assertion is to check that we don't sync allocs of size 1
    // (there's never a need, so sync_point() shouldn't have been called)
    ASSERT(((trailing != ref) || from_reduce()) &&
           "Missized allocation in thread sync calc");
    return trailing;
  };

  // 2. find sync sizes (shared, global etc)
  lt.walk([&](LoopTree::TreeRef ref, int) {
    if (lt.kind(ref) == LoopTree::LOOP) {
      return;
    }
    auto node_ref = lt.node(ref);
    for (auto inp : lt.ir.node(node_ref).inputs()) {
      auto num_threads = alloc_threads.at(inp);
      if (num_threads <= 1) {
        continue;
      }
      auto sync = sync_point(aux.allocs.at(inp), ref);
      if (num_threads > cuda_aux.threads_per_block) {
        syncs[sync] = sync_global;
      } else if (num_threads > cuda_aux.threads_per_warp) {
        syncs[sync] = sync_shared;
      } else {
        syncs[sync] = sync_warp;
      }
    }
  });

  cuda_aux.alloc_threads = alloc_threads;
  cuda_aux.syncs = syncs;
}

CudaAux calc_cuda_aux(const LoopTree &lt, const Auxiliary &aux,
                      const std::unordered_set<LoopTree::TreeRef> &threaded_) {
  CUDA_SAFE_CALL(cuInit(0));
  CudaAux cuda_aux;
  auto threaded = threaded_;
  if (threaded.size() == 1 && threaded.count(-1)) {
    threaded.clear();
    lt.walk([&](LoopTree::TreeRef ref, int) {
      if (trivially_parallel(lt, ref)) {
        threaded.insert(ref);
      }
    });
  } else {
    for (auto ref : threaded) {
      ASSERT(trivially_parallel(lt, ref) &&
             "Loop not yet threadable! TODO: warp-level reductions");
    }
  }
  lt.walk([&](LoopTree::TreeRef ref, int) {
    if (lt.kind(ref) != LoopTree::NODE) {
      return;
    }
    auto parent = lt.parent(ref);
    auto inner = 1;
    while (parent != -1) {
      ASSERT(lt.kind(parent) == LoopTree::LOOP);
      if (threaded.count(parent)) {
        if (cuda_aux.threaded.count(parent)) {
          auto alt_inner = cuda_aux.threaded.at(parent);
          inner = std::max(inner, alt_inner);
          // self consistency
          // ASSERT((alt_inner == -1 || alt_inner == inner)) <<
          //       "Found mismatched threading strategy for " <<
          //			 lt.ir.var(lt.loop(parent).var).name() <<
          //			 " size: " << alt_inner << " vs " << inner;
        }
        cuda_aux.threaded[parent] = inner;
        inner *= lt.loop(parent).size;
      }
      parent = lt.parent(parent);
    }
    cuda_aux.threaded[-1] = std::max(inner, cuda_aux.threaded[-1]);
  });
  unroll(lt, cuda_aux);
  // TODO multiple devices
  CUdevice cuDevice;
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  // TODO Y, Z thread scheduling
  CUDA_SAFE_CALL(cuDeviceGetAttribute(&cuda_aux.threads_per_block,
                                      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                      cuDevice));
  CUDA_SAFE_CALL(cuDeviceGetAttribute(&cuda_aux.threads_per_warp,
                                      CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice));
  gen_threading_info(lt, aux, cuda_aux);
  return cuda_aux;
}

/*
TODO: this logic is not yet implemented -- runs checks instead

 Compute might not fit into a single kernel for various reasons
 1. if shared dynamic memory that needs syncing > shared memory
 2. if threading forces global syncs

 In these cases, we need to identify an outer CPU loop tree.

 e.g. "---" denotes kernel split, indentation at what level

  for a: L0
    compute
  ---
  for b: L1
    for a:
      compute
  for c:
    compute
  ---
  for c: L2 <-- CPU loop
    for a: L3 <- kernel
      compute
    ---

 will become

 cuda_exec(L0)
 cuda_exec(L1)
 for c: L2
   cuda_exec(L3)

*/
bool needs_multikernel_support(const LoopTree &lt, const Auxiliary &aux,
                               const CudaAux &cuda_aux) {
  bool needs_multikernel = false;
  lt.walk([&](LoopTree::TreeRef ref, int) {
    if (cuda_aux.syncs.count(ref)) {
      if (cuda_aux.syncs.at(ref) == sync_global) {
        needs_multikernel = true;
      }
      if (count_threads(lt, cuda_aux, ref) > cuda_aux.threads_per_block) {
        needs_multikernel = true;
      }
    }
  });
  return needs_multikernel;
}

// returns a cuda string and dispatch params (blocks, threads)
std::pair<std::string, std::pair<size_t, size_t>> cuda_code_and_dispatch(
    const LoopTree &lt, const std::unordered_set<LoopTree::TreeRef> &threaded) {
  auto aux = calculate_aux(lt);
  auto cuda_aux = calc_cuda_aux(lt, aux, threaded);
  ASSERT(!needs_multikernel_support(lt, aux, cuda_aux))
      << "This parameterization needs multiple kernels, which is not yet "
         "supported";
  auto cuda_code = cuda_compile(lt, cuda_aux);

  size_t needed_threads = thread_scope(lt, cuda_aux, -1);
  size_t num_blocks = (needed_threads + cuda_aux.threads_per_block - 1) /
                      cuda_aux.threads_per_block;
  size_t num_threads =
      std::min(needed_threads, (size_t)cuda_aux.threads_per_block);

  return std::make_pair(cuda_code, std::make_pair(num_blocks, num_threads));
}

int availableCudaGPUs() {
  CUDA_SAFE_CALL(cuInit(0));
  int avail;
  CUDA_SAFE_CALL(cuDeviceGetCount(&avail));
  return avail;
}

struct CudaGPUHardware : public Hardware {
  CudaGPUHardware() : Hardware("cuda", availableCudaGPUs()) {}

  Memory alloc(size_t size) override {
    void *ptr = nullptr;
    auto err = cudaMallocManaged(&ptr, size);
    gpuErrchk(err);
    return Memory{0x1 | 1 << id_, ptr};
  }

  void free(Memory &data) override {
    cudaFree(data.address);
    data.address = nullptr;
    data.compatible = 0;
  }
  static Hardware *create() { return new CudaGPUHardware(); }
};

struct CudaCompiled : public Compiled {
  char *ptx;
  CUfunction kernel;
  std::string code;
  size_t num_blocks = 0;
  size_t num_threads = 0;

  size_t peak_bandwidth_gb = 0;

  CUcontext context;
  CUmodule module;
  CUdevice cuDevice;

  CudaCompiled(const LoopTree &lt,
               const std::unordered_set<LoopTree::TreeRef> &threaded,
               LoopTree::TreeRef ref) {
    auto cc = cuda_code_and_dispatch(lt, threaded);
    code = cc.first;
    num_blocks = cc.second.first;
    num_threads = cc.second.second;

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,         // prog
                                       code.c_str(),  // buffer
                                       "kernel.cu",   // name
                                       0,             // numHeaders
                                       NULL,          // headers
                                       NULL));        // includeNames
    const char *opts[] = {
        //"--extra-device-vectorization"
        //"--gpu-architecture=compute_60",
        //"--generate-line-info"
    };
    nvrtcResult compileResult = nvrtcCompileProgram(prog,   // prog
                                                    0,      // numOptions
                                                    opts);  // options
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    ASSERT(compileResult == NVRTC_SUCCESS) << log;
    delete[] log;

    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "kernel"));

    int memory_clock;
    int memory_bus_width;
    CUDA_SAFE_CALL(cuDeviceGetAttribute(
        &memory_clock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, cuDevice));
    CUDA_SAFE_CALL(cuDeviceGetAttribute(
        &memory_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        cuDevice));

    peak_bandwidth_gb =
        2 * ((long)memory_clock * (long)memory_bus_width) / 1e6 / 8;
    int_properties["bandwidth"] = peak_bandwidth_gb;
    string_properties["code"] = code;
  }

  ~CudaCompiled() {
    CUDA_SAFE_CALL(cuCtxDestroy(context));
    free(ptx);
  }

  void run(const std::vector<void *> &memory, bool sync) const override {
    std::vector<void *> mem;
    for (auto &v : memory) {
      mem.emplace_back(reinterpret_cast<void *>(const_cast<void **>(&v)));
    }
    void **args = mem.data();
    CUDA_SAFE_CALL(cuLaunchKernel(kernel, num_blocks, 1, 1,  // grid dim
                                  num_threads, 1, 1,         // block dim
                                  0, NULL,    // shared mem and stream
                                  args, 0));  // arguments
    if (sync) {
      CUDA_SAFE_CALL(cuCtxSynchronize());
    }
  }
};

struct CudaBackend : public Backend {
  CudaBackend() : Backend("cuda") {}

  std::unique_ptr<Compiled> compile_impl(
      const LoopTree &lt, const std::unordered_set<LoopTree::TreeRef> &parallel,
      LoopTree::TreeRef root) override {
    return std::make_unique<CudaCompiled>(lt, parallel, root);
  }

  int hardware_requirement() const override {
    for (auto &hw : getHardware()) {
      if (hw->name() == "cuda") {
        return 1 << hw->id();
      }
    }
    ASSERT(0) << "Tried to register Cuda backend but couldn't find Cuda "
                 "hardware registration";
    return 0;
  }
};

static RegisterHardware cuda_hw_reg_(std::make_shared<CudaGPUHardware>());
static RegisterBackend cuda_backend_reg_(std::make_shared<CudaBackend>());

}  // namespace loop_tool
