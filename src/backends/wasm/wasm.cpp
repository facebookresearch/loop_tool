/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/wasm.h"

using namespace loop_tool;
using namespace symbolic;

WebAssemblyCompiler::WebAssemblyCompiler(const LoopTree& lt)
    : Compiler(lt), cg(std::make_shared<wasmblr::CodeGenerator>()) {
  // Logic to calculate "local storage" opportunities.  This effectively means
  // "in-register" and requires static information about things like vector
  // lanes and which register data lives in
  // TODO this is overly restrictive -- we can also store things in register
  // if only their relevant vars are unrolled (and irrelevant are not).
  auto completely_unrolled = [&](LoopTree::TreeRef ref, LoopTree::TreeRef lca,
                                 const std::unordered_set<IR::VarRef>& vars) {
    ref = lt.parent(ref);
    while (ref != lca) {
      const auto& annot = lt.annotation(ref);
      const auto& loop = lt.loop(ref);
      if (vars.count(loop.var) && (annot != "unroll") &&
          (annot != "vectorize")) {
        return false;
      }
      ref = lt.parent(ref);
    }
    return true;
  };

  lt.walk([&](LoopTree::TreeRef ref, int depth) {
    if (lt.kind(ref) == LoopTree::NODE) {
      return;
    }
    if (should_vectorize(ref)) {
      vectorized_loops.insert(ref);
    }
  });

  const auto& nodes = lt.ir.nodes();
  for (auto i = 0; i < nodes.size(); ++i) {
    const auto& node_ref = nodes.at(i);
    const auto& node = lt.ir.node(node_ref);
    // forced to use real memory in these cases
    if (!lt.scheduled.count(node_ref) || node.op() == Operation::write) {
      continue;
    }
    const auto& alloc = allocations.at(node_ref);
    bool scheduled_consumers = true;
    auto ref = lt.scheduled.at(node_ref);
    auto vars = to_set(node.vars());
    bool unrolled = completely_unrolled(ref, alloc.lca, vars);
    for (const auto& consumer_ref : node.outputs()) {
      if (!lt.scheduled.count(consumer_ref)) {
        scheduled_consumers = false;
        break;
      }
      if (!completely_unrolled(lt.scheduled.at(consumer_ref), alloc.lca,
                               vars)) {
        unrolled = false;
        break;
      }
    }
    // we cannot address this memory statically (will need runtime
    // information)
    if (!scheduled_consumers || (!unrolled && alloc.size() > 1)) {
      continue;
    }

    auto store_on_stack = should_store_stack(node_ref);
    auto vectorized_dim = should_store_vectorized_dim(node_ref);
    auto store_vector = (vectorized_dim != -1);

    if (store_on_stack && store_vector) {
      stack_vector_storage[node_ref] = vectorized_dim;
    } else if (store_on_stack) {
      stack_storage.insert(node_ref);
    } else if (store_vector) {
      local_vector_storage[node_ref] = vectorized_dim;
    } else {
      local_storage.insert(node_ref);
    }
  }
}

int64_t WebAssemblyCompiler::get_unroll_offset(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    const std::unordered_map<LoopTree::TreeRef, int32_t>& unrolls) const {
  auto access = gen_access(node_ref, ref);
  const auto& idx_expr = get_scoped_expr(access);
  return get_unroll_offset(node_ref, ref, access.alloc.lca, idx_expr, unrolls);
}

int64_t WebAssemblyCompiler::get_unroll_offset(
    IR::NodeRef node_ref, LoopTree::TreeRef ref, LoopTree::TreeRef root,
    const Expr& idx_expr,
    const std::unordered_map<LoopTree::TreeRef, int32_t>& unrolls) const {
  const auto& vars = to_set(lt.ir.node(node_ref).vars());
  auto p = lt.parent(ref);
  int64_t offset = 0;
  while (p != root) {
    auto stride = inner_sizes.at(p);
    const auto& loop = lt.loop(p);
    auto sym = var_to_sym.at(loop.var);
    auto var_stride = differentiate(idx_expr, sym).simplify().evaluate();
    if (unrolls.count(p) && vars.count(lt.loop(p).var)) {
      offset += unrolls.at(p) * stride * var_stride;
    }
    p = lt.parent(p);
  }
  return offset;
}

void WebAssemblyCompiler::push_expr_to_stack(
    const Expr& idx_expr,
    std::unordered_map<symbolic::Symbol,
                       std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                       symbolic::Hash<symbolic::Symbol>>
        sym_strides,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls,
    int32_t base_stride) const {
  bool emitted = false;
  auto idx_expr_offset = intercept(idx_expr).evaluate();
  if (idx_expr_offset) {
    cg->i32.const_(idx_expr_offset * base_stride);
    emitted = true;
  }
  for (const auto& sym : idx_expr.symbols()) {
    auto stride_expr = differentiate(idx_expr, sym).simplify();
    ASSERT(stride_expr.can_evaluate()) << "Invalid indexing expr";
    auto stride = stride_expr.evaluate();
    if (stride == 0) {
      continue;
    }
    // each symbol can have multiple iterators
    for (const auto& p : sym_strides.at(sym)) {
      // float size
      int32_t inner_stride = p.second * stride * base_stride;
      if (inner_stride == 0) {
        continue;
      }
      if (unrolls.count(p.first)) {
        continue;
      }
      cg->local.get(iterators.at(p.first));
      if (inner_stride > 1) {
        cg->i32.const_(inner_stride);
        cg->i32.mul();
      }
      if (emitted) {
        cg->i32.add();
      } else {
        emitted = true;
      }
    }
  }
  if (!emitted) {
    cg->i32.const_(0);
  }
}

int32_t WebAssemblyCompiler::push_access_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  auto access = gen_access(node_ref, ref);
  const auto& idx_expr = get_scoped_expr(access);
  // grab the relevant loops
  auto sym_strides = get_symbol_strides(ref, access.alloc.lca);
  // memory needs 4x for bytes sizeof(float)
  int32_t offset =
      get_unroll_offset(node_ref, ref, access.alloc.lca, idx_expr, unrolls) * 4;

  push_expr_to_stack(idx_expr, sym_strides, unrolls, 4);
  return offset;
}

bool WebAssemblyCompiler::push_constraints_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  auto acc = gen_access(node_ref, ref);
  const auto& constraints = node_ref != lt.node(ref)
                                ? get_constraints(acc)
                                : std::vector<std::pair<Expr, int64_t>>{};
  if (!constraints.size()) {
    return false;
  };
  auto sym_strides = get_symbol_strides(ref, acc.alloc.lca);
  bool emitted = false;
  for (const auto& c : constraints) {
    const auto& idx_expr = c.first;
    const auto lower_bound = intercept(idx_expr).evaluate() < 0;
    const auto upper_bound = c.second != -1;

    if (!lower_bound && !upper_bound) {
      continue;
    }
    if (lower_bound) {
      cg->i32.const_(0);
      push_expr_to_stack(idx_expr, sym_strides, unrolls, 1);
      if (upper_bound) {
        cg->local.tee(get_tmp_i32());
      }
      cg->i32.le_s();
      if (emitted) {
        cg->i32.and_();
      } else {
        emitted = true;
      }
    }
    if (upper_bound) {
      cg->i32.const_(c.second);
      if (lower_bound) {
        cg->local.get(get_tmp_i32());
      } else {
        push_expr_to_stack(idx_expr, sym_strides, unrolls, 1);
      }
      cg->i32.gt_s();
      if (emitted) {
        cg->i32.and_();
      } else {
        emitted = true;
      }
    }
  }
  // degenerate case
  if (!emitted) {
    return false;
  }
  cg->if_(cg->f32);
  return true;
}

void WebAssemblyCompiler::push_float_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls,
    bool force_memory_load) const {
  bool constrained = push_constraints_to_stack(node_ref, ref, unrolls);
  if (!force_memory_load && stack_f32.count(node_ref)) {
    // it's on the stack
  } else if (!force_memory_load && local_f32.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_f32.at(node_ref);
    ASSERT(off < locals.size());
    cg->local.get(locals.at(off));
  } else if (!force_memory_load && local_v128.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_v128.at(node_ref);
    const auto& var = local_vector_storage.at(node_ref);
    const auto& vs = lt.ir.node(node_ref).vars();
    auto broadcast = !(vs.size() && vs.back() == var);
    if (broadcast) {
      ASSERT((off) < locals.size());
      cg->local.get(locals.at(off));
      cg->v128.f32x4_extract_lane(0);
    } else {
      ASSERT((off / 4) < locals.size());
      cg->local.get(locals.at(off / 4));
      cg->v128.f32x4_extract_lane(off % 4);
    }
  } else if (!force_memory_load && stack_v128.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    ASSERT(off < 4);
    cg->v128.f32x4_extract_lane(off);
  } else {
    auto offset = push_access_to_stack(node_ref, ref, unrolls);
    cg->f32.load(0, memory_locations.at(resolved_reads.at(node_ref)) + offset);
  }
  if (constrained) {
    cg->else_();
    cg->f32.const_(0);
    cg->end();
  }
}

void WebAssemblyCompiler::store_float_from_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  if (stack_storage.count(node_ref)) {
    // pass
  } else if (local_storage.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_f32.at(node_ref);
    ASSERT(off < locals.size());
    cg->local.set(locals.at(off));
    // TODO these both assume broadcast (incorrectly)
  } else if (stack_vector_storage.count(node_ref)) {
    cg->v128.f32x4_splat();
  } else if (local_vector_storage.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_v128.at(node_ref);
    const auto& var = local_vector_storage.at(node_ref);
    const auto& vs = lt.ir.node(node_ref).vars();
    auto broadcast = !(vs.size() && vs.back() == var);
    if (broadcast) {
      ASSERT(off < locals.size());
      cg->v128.f32x4_splat();
      cg->local.set(locals.at(off));
    } else {
      ASSERT((off / 4) < locals.size());
      cg->local.set(get_tmp_f32());
      cg->local.get(locals.at(off / 4));
      cg->local.get(get_tmp_f32());
      cg->v128.f32x4_replace_lane(off % 4);
      cg->local.set(locals.at(off / 4));
    }
  } else {
    cg->local.set(get_tmp_f32());
    auto store_offset = push_access_to_stack(node_ref, ref, unrolls);
    cg->local.get(get_tmp_f32());
    cg->f32.store(
        0, memory_locations.at(resolved_writes.at(node_ref)) + store_offset);
  }
}

void WebAssemblyCompiler::store_vector_from_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls,
    IR::VarRef dim) const {
  if (stack_vector_storage.count(node_ref)) {
    // pass
  } else if (local_vector_storage.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_v128.at(node_ref);
    const auto& var = local_vector_storage.at(node_ref);
    const auto& vs = lt.ir.node(node_ref).vars();
    auto broadcast = !(vs.size() && vs.back() == var);
    if (broadcast) {
      ASSERT((off) < locals.size());
      cg->local.set(locals.at(off));
    } else {
      ASSERT((off / 4) < locals.size());
      cg->local.set(locals.at(off / 4));
    }
  } else if (local_storage.count(node_ref)) {
    // get associated lanes
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_f32.at(node_ref);
    ASSERT(off + 3 < locals.size());
    cg->local.set(get_tmp_v128());
    for (auto i = 0; i < 4; ++i) {
      cg->local.get(get_tmp_v128());
      cg->v128.f32x4_extract_lane(i);
      cg->local.set(locals.at(off + i));
    }
  } else {
    ASSERT(!stack_storage.count(node_ref));
    cg->local.set(get_tmp_v128());
    auto store_offset = push_access_to_stack(node_ref, ref, unrolls);
    cg->local.get(get_tmp_v128());
    cg->v128.store(
        0, memory_locations.at(resolved_writes.at(node_ref)) + store_offset);
  }
}

void WebAssemblyCompiler::push_vector_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls, IR::VarRef dim,
    bool force_memory_load) const {
  const auto& vars = to_set(lt.ir.node(node_ref).vars());
  bool broadcast = !vars.count(dim);
  ASSERT(broadcast || lt.ir.node(node_ref).vars().back() == dim)
      << "gather not yet implemented";
  if (!force_memory_load && stack_f32.count(node_ref)) {
    ASSERT(broadcast) << "cannot vector load from f32 on stack";
  }
  if (!force_memory_load && stack_v128.count(node_ref)) {
    return;
  } else if (!force_memory_load && local_v128.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_v128.at(node_ref);
    const auto& var = local_vector_storage.at(node_ref);
    const auto& vs = lt.ir.node(node_ref).vars();
    auto broadcast = !(vs.size() && vs.back() == var);
    if (broadcast) {
      ASSERT(off < locals.size());
      cg->local.get(locals.at(off));
    } else {
      ASSERT((off / 4) < locals.size());
      cg->local.get(locals.at(off / 4));
    }
  } else if (!force_memory_load && local_f32.count(node_ref)) {
    ASSERT(0) << "TODO lane based vector load (gather) from locals";
    auto parent_loop = lt.parent(ref);
    auto base_unroll = unrolls.at(parent_loop);
    for (auto lane = 0; lane < 4; ++lane) {
      unrolls[parent_loop] = base_unroll + lane;
      const auto off = get_unroll_offset(node_ref, ref, unrolls);
      const auto& locals = local_f32.at(node_ref);
      ASSERT(off < locals.size());
      cg->local.get(locals.at(off));
    }
  } else if (broadcast &&
             (stack_f32.count(node_ref) || local_f32.count(node_ref))) {
    push_float_to_stack(node_ref, ref, unrolls, force_memory_load);
    cg->v128.f32x4_splat();
  } else if (broadcast) {
    auto offset = push_access_to_stack(node_ref, ref, unrolls);
    cg->v128.load32_splat(
        0, memory_locations.at(resolved_reads.at(node_ref)) + offset);
  } else {  // load directly from memory
    auto offset = push_access_to_stack(node_ref, ref, unrolls);
    cg->v128.load(0, memory_locations.at(resolved_reads.at(node_ref)) + offset);
  }
}

void WebAssemblyCompiler::emit_vectorized_node(
    LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  const auto& parent = lt.parent(ref);
  if (!unrolls.count(parent)) {
    return emit_node(ref, unrolls);
  }
  // ASSERT(parent != -1) << "cannot vectorize a root node";
  // ASSERT(ref != -1) << "cannot vectorize the root node";
  const auto& loop = lt.loop(parent);
  const auto& node_ref = lt.node(ref);
  const auto& node = lt.ir.node(node_ref);
  if (lt.children(lt.parent(ref)).at(0) == ref) {
    emit_reset(lt.parent(ref));
  }

  for (const auto& inp_ref : node.inputs()) {
    push_vector_to_stack(inp_ref, ref, unrolls, loop.var);
  }
  bool is_reduction = lt.ir.reduction_vars(node_ref).size();
  if (is_reduction) {
    push_vector_to_stack(node_ref, ref, unrolls, loop.var);
  }
  if (node.op() == Operation::read) {
    push_vector_to_stack(node_ref, ref, unrolls, loop.var, true);
  }

  switch (node.op()) {
    case Operation::add:
      cg->v128.f32x4_add();
      break;
    case Operation::subtract:
      cg->v128.f32x4_sub();
      break;
    case Operation::multiply:
      cg->v128.f32x4_mul();
      break;
    case Operation::divide:
      cg->v128.f32x4_div();
      break;
    case Operation::sqrt:
      cg->v128.f32x4_sqrt();
      break;
    case Operation::min:
      cg->v128.f32x4_min();
      break;
    case Operation::max:
      cg->v128.f32x4_max();
      break;
    case Operation::negate:
      cg->v128.f32x4_neg();
      break;
    case Operation::abs:
      cg->v128.f32x4_abs();
      break;
    case Operation::copy:
    case Operation::read:
    case Operation::write:
      break;
    default:
      ASSERT(0) << "Can't handle op yet for wasm " << lt.ir.dump(node_ref);
  };

  store_vector_from_stack(node_ref, ref, unrolls, loop.var);
}

void WebAssemblyCompiler::emit_vectorized_loop(
    LoopTree::TreeRef ref, std::unordered_map<IR::VarRef, int> overrides,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  const auto& loop = lt.loop(ref);
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

  ASSERT(tail == 0) << "invalid vectorization scheme proposed";

  // generate any resets
  if (lt.children(lt.parent(ref)).at(0) == ref) {
    emit_reset(lt.parent(ref));
  }

  if (size == 4) {     // genuine vectorization
    unrolls[ref] = 0;  // to get "push_access_to_stack" working cleanly
    for (auto c : lt.children(ref)) {
      emit_vectorized_node(c, unrolls);
    }
  } else {  // unrolled version
    for (auto i = 0; i < size; ++i) {
      unrolls[ref] = i;
      for (auto c : lt.children(ref)) {
        emit(c, overrides, unrolls);
      }
    }
  }
}

void WebAssemblyCompiler::emit_node(
    LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  const auto& node_ref = lt.node(ref);
  const auto& node = lt.ir.node(node_ref);

  if (lt.children(lt.parent(ref)).at(0) == ref) {
    emit_reset(lt.parent(ref));
  }

  for (const auto& inp_ref : node.inputs()) {
    push_float_to_stack(inp_ref, ref, unrolls);
  }
  bool is_reduction = lt.ir.reduction_vars(node_ref).size();
  if (is_reduction) {
    push_float_to_stack(node_ref, ref, unrolls);
  }
  if (node.op() == Operation::read) {
    push_float_to_stack(node_ref, ref, unrolls, true);
  }

  switch (node.op()) {
    case Operation::add:
      cg->f32.add();
      break;
    case Operation::subtract:
      cg->f32.sub();
      break;
    case Operation::multiply:
      cg->f32.mul();
      break;
    case Operation::divide:
      cg->f32.div();
      break;
    case Operation::sqrt:
      cg->f32.sqrt();
      break;
    case Operation::min:
      cg->f32.min();
      break;
    case Operation::max:
      cg->f32.max();
      break;
    case Operation::negate:
      cg->f32.neg();
      break;
    case Operation::abs:
      cg->f32.abs();
      break;
    case Operation::copy:
    case Operation::read:
    case Operation::write:
      break;
    default:
      ASSERT(0) << "Can't handle op yet for wasm " << lt.ir.dump(node_ref);
  };

  store_float_from_stack(node_ref, ref, unrolls);
}

bool WebAssemblyCompiler::needs_reset(IR::NodeRef node_ref) const {
  const auto& node = lt.ir.node(node_ref);
  bool needs_set =
      lt.ir.reduction_vars(node_ref).size() && node.op() != Operation::view;
  for (const auto& input : node.inputs()) {
    // this is only necessary if the view has "empty" outputs
    // which only happens if its access is bounded
    if (lt.ir.node(input).op() == Operation::view &&
        !lt.scheduled.count(input)) {
      const auto& acc = gen_access(input, lt.scheduled.at(node_ref));
      for (const auto& b : acc.bounds) {
        if (b.first != 0 || b.second != -1) {
          needs_set = true;
        }
      }
    }
  }
  return needs_set;
}

void WebAssemblyCompiler::emit_reset(LoopTree::TreeRef ref) const {
  auto value = [&](const Node& node) -> float {
    if (node.op() == Operation::add) {
      return 0;
    } else if (node.op() == Operation::multiply) {
      return 1;
    } else if (node.op() == Operation::max) {
      return std::numeric_limits<float>::lowest();
    } else if (node.op() == Operation::min) {
      return std::numeric_limits<float>::max();
    } else if (node.op() == Operation::write) {
      return 0;  // TODO fix
    } else if (node.op() == Operation::view) {
      return 0;  // TODO fix
    }
    ASSERT(0) << "cannot find default value for " << dump(node.op());
    return -1;
  };
  for (const auto& p : allocations) {
    const auto& alloc = p.second;
    if (alloc.lca != ref) {
      continue;
    }

    const auto& node = lt.ir.node(alloc.node_ref);
    if (!lt.scheduled.count(alloc.node_ref) || !needs_reset(alloc.node_ref)) {
      continue;
    }
    if (stack_f32.count(alloc.node_ref)) {
      // cg->f32.const_(value(node));
      // ASSERT(0) << "reset not possible for stack resident memory";
    } else if (local_f32.count(alloc.node_ref)) {
      for (const auto& local : local_f32.at(alloc.node_ref)) {
        cg->f32.const_(value(node));
        cg->local.set(local);
      }
    } else if (local_v128.count(alloc.node_ref)) {
      for (const auto& local : local_v128.at(alloc.node_ref)) {
        cg->f32.const_(value(node));
        cg->v128.f32x4_splat();
        cg->local.set(local);
      }
    } else {
      auto iter = cg->local(cg->i32);
      cg->i32.const_(0);
      cg->local.set(iter);
      cg->loop(cg->void_);

      cg->local.get(iter);
      cg->f32.const_(value(node));
      cg->f32.store(0, memory_locations.at(resolved_writes.at(alloc.node_ref)));

      cg->local.get(iter);
      cg->i32.const_(4);
      cg->i32.add();
      cg->local.tee(iter);
      cg->i32.const_(alloc.size() * 4);
      cg->i32.lt_u();
      cg->br_if(0);
      cg->end();
    }
  }
}

// loops need to be exactly 4 wide
// inputs/outputs need to be fully (4 wide) addressable
bool WebAssemblyCompiler::should_vectorize(LoopTree::TreeRef ref) const {
  // we can only vectorize unrolled size 4 loops composed of compute nodes
  if (lt.kind(ref) != LoopTree::LOOP) {
    return false;
  }
  if (lt.annotation(ref) != "vectorize") {
    return false;
  }
  auto loop = lt.loop(ref);
  if (loop.size != 4 || loop.tail != 0) {
    return false;
  }
  // check if the children are all "nodes" rather than nested loops
  auto children = lt.children(ref);
  for (auto c : children) {
    if (lt.kind(c) == LoopTree::LOOP) {
      return false;
    }
    auto node_ref = lt.node(c);
    const auto& node = lt.ir.node(node_ref);
    // TODO deal with view logic in vectorized way
    if (node.op() == Operation::view) {
      return false;
    }
    // TODO We can't vectorize reductions (for now)
    for (auto v : lt.ir.reduction_vars(node_ref)) {
      if (v == loop.var) {
        return false;
      }
    }
  }
  return true;
}

bool WebAssemblyCompiler::should_store_stack(IR::NodeRef node_ref) const {
  if (!lt.scheduled.count(node_ref)) {
    return false;
  }
  const auto& node = lt.ir.node(node_ref);
  if (node.outputs().size() != 1) {
    return false;
  }
  auto consumer = node.outputs().at(0);
  if (!lt.scheduled.count(consumer)) {
    return false;
  }
  auto ref = lt.scheduled.at(node_ref);
  auto consumer_ref = lt.scheduled.at(consumer);
  const auto& children = lt.children(lt.parent(consumer_ref));
  bool valid = false;
  for (auto i = 1; i < children.size(); ++i) {
    if (children.at(i) == consumer_ref) {
      if (children.at(i - 1) != ref) {
        return false;
      } else {
        valid = true;
      }
    }
  }
  return valid;
}

// inputs to a vectorized loop should store to a vector register if possible
IR::VarRef WebAssemblyCompiler::should_store_vectorized_dim(
    IR::NodeRef node_ref) const {
  // assume that we should only store vectorized if all consumers are vectorized
  const auto& node = lt.ir.node(node_ref);
  if (node.outputs().size() == 0) {
    return false;
  }
  IR::VarRef vectorized_var = -1;
  for (auto o : node.outputs()) {
    if (!lt.scheduled.count(o)) {
      return -1;
    }
    auto o_ref = lt.parent(lt.scheduled.at(o));
    if (!vectorized_loops.count(o_ref)) {
      return -1;
    }
    if (vectorized_var == -1) {
      // guaranteed valid because loop is vectorized
      vectorized_var = lt.loop(o_ref).var;
    } else if (vectorized_var != lt.loop(lt.parent(o_ref)).var) {
      return -1;  // multiple vars to vectorize over
    }
  }
  return vectorized_var;
}

void WebAssemblyCompiler::emit_loop(
    LoopTree::TreeRef ref, std::unordered_map<IR::VarRef, int> overrides,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  if (vectorized_loops.count(ref)) {
    emit_vectorized_loop(ref, overrides, unrolls);
    return;
  }

  const auto& loop = lt.loop(ref);
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

  // generate any resets
  if (lt.children(lt.parent(ref)).at(0) == ref) {
    emit_reset(lt.parent(ref));
  }

  // two cases
  if (lt.annotation(ref) == "unroll" || size == 1) {  // 1. unrolled loop
    for (auto i = 0; i < size; ++i) {
      unrolls[ref] = i;
      for (auto c : lt.children(ref)) {
        emit(c, overrides, unrolls);
      }
    }
  } else {  // 2. default loop
    // generate any loop header logic
    auto iter = cg->local(cg->i32);
    iterators[ref] = iter;
    cg->i32.const_(0);
    cg->local.set(iter);
    cg->loop(cg->void_);

    // generate body code
    for (auto c : lt.children(ref)) {
      emit(c, overrides, unrolls);
    }

    // generate any loop footer logic
    cg->local.get(iter);
    cg->i32.const_(1);
    cg->i32.add();
    cg->local.tee(iter);
    cg->i32.const_(size);
    cg->i32.lt_u();
    cg->br_if(0);
    cg->end();
  }

  // generate any tail logic
  if (tail > 0) {
    overrides[loop.var] = tail;
    if (lt.annotation(ref) == "unroll" || size == 1) {
      unrolls[ref] = size;
    }
    // value is now fixed
    for (auto c : lt.children(ref)) {
      emit(c, overrides, unrolls);
    }
  }
}

int32_t WebAssemblyCompiler::get_tmp_i32() const {
  if (tmp_i32 == -1) {
    tmp_i32 = cg->local(cg->i32);
  }
  return tmp_i32;
}

int32_t WebAssemblyCompiler::get_tmp_f32() const {
  if (tmp_f32 == -1) {
    tmp_f32 = cg->local(cg->f32);
  }
  return tmp_f32;
}

int32_t WebAssemblyCompiler::get_tmp_v128() const {
  if (tmp_v128 == -1) {
    tmp_v128 = cg->local(cg->v128);
  }
  return tmp_v128;
}

void WebAssemblyCompiler::emit(
    LoopTree::TreeRef ref, std::unordered_map<IR::VarRef, int> overrides,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  if (ref == -1) {
    auto func = cg->function({}, {}, [&]() {
      for (const auto& node_ref : local_storage) {
        const auto& alloc = allocations.at(node_ref);
        for (auto i = 0; i < alloc.size(); i++) {
          local_f32[node_ref].emplace_back(cg->local(cg->f32));
        }
      }
      for (const auto& node_ref : stack_storage) {
        stack_f32.insert(node_ref);
      }
      for (const auto& p : local_vector_storage) {
        const auto& node_ref = p.first;
        const auto& var = p.second;
        const auto& alloc = allocations.at(node_ref);
        const auto& vs = lt.ir.node(node_ref).vars();
        auto broadcast = !(vs.size() && vs.back() == var);
        auto num = broadcast ? alloc.size() : ((alloc.size() + 3) / 4);
        for (auto i = 0; i < num; i++) {
          local_v128[node_ref].emplace_back(cg->local(cg->v128));
        }
      }
      for (const auto& p : stack_vector_storage) {
        auto node_ref = p.first;
        stack_v128.insert(node_ref);
      }
      for (auto c : lt.roots) {
        emit(c, overrides, unrolls);
      }
    });
    cg->export_(func, "fn");
  } else if (lt.kind(ref) == LoopTree::NODE) {
    emit_node(ref, unrolls);
  } else {
    emit_loop(ref, overrides, unrolls);
  }
}

std::vector<uint8_t> WebAssemblyCompiler::emit() const {
  cg = std::move(std::make_shared<wasmblr::CodeGenerator>());
  tmp_i32 = -1;
  stack_f32.clear();
  local_f32.clear();
  tmp_f32 = -1;
  stack_v128.clear();
  local_v128.clear();
  tmp_v128 = -1;
  memory_locations.clear();
  iterators.clear();

  std::vector<std::pair<IR::NodeRef, int64_t>> sizes(allocations.size());
  for (const auto& p : allocations) {
    sizes[p.second.mem_idx] = std::make_pair(p.first, p.second.size());
  }
  int32_t running_location = 0;
  for (const auto& p : sizes) {
    memory_locations[p.first] = running_location;
    running_location += p.second * 4;
  }
  auto pages = running_location / (1 << 16) + 1;
  cg->memory(pages).export_("mem");
  emit(-1, {}, {});

  return cg->emit();
}
