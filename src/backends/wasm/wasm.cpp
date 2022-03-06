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
  auto completely_unrolled = [&](LoopTree::TreeRef ref, LoopTree::TreeRef lca) {
    ref = lt.parent(ref);
    while (ref != lca) {
      if (lt.annotation(ref) != "unroll") {
        return false;
      }
      ref = lt.parent(ref);
    }
    return true;
  };
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
    bool unrolled = true;
    for (const auto& consumer_ref : node.outputs()) {
      if (!lt.scheduled.count(consumer_ref)) {
        scheduled_consumers = false;
        break;
      }
      if (!completely_unrolled(lt.scheduled.at(consumer_ref), alloc.lca)) {
        unrolled = false;
        break;
      }
    }
    // we cannot address this memory statically (will need runtime
    // information)
    if (!scheduled_consumers || !unrolled) {
      continue;
    }

    const auto& peak_next = i < nodes.size() - 1 ? nodes.at(i + 1) : -1;
    if (!needs_reset(alloc.node_ref) && alloc.size() == 1 &&
        node.outputs().size() == 1 && node.outputs().at(0) == peak_next) {
      stack_storage.insert(node_ref);
    } else {
      local_storage.insert(node_ref);
    }
  }
}

Expr WebAssemblyCompiler::get_scoped_expr(
    const Compiler::Access& access) const {
  Expr full_expr(0);
  for (auto i = 0; i < access.scoped_exprs.size(); ++i) {
    auto stride = access.alloc.size(i + 1);
    const auto& expr = access.scoped_exprs.at(i);
    full_expr = full_expr + expr * Expr(stride);
  }
  full_expr = full_expr.simplify();
  return full_expr;
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

// returns constant offset for all symbols
std::unordered_map<Symbol, std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                   Hash<Symbol>>
WebAssemblyCompiler::get_symbol_strides(
    LoopTree::TreeRef ref, LoopTree::TreeRef root,
    const std::unordered_map<LoopTree::TreeRef, int32_t>& unrolls) const {
  std::unordered_map<Symbol, std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                     Hash<Symbol>>
      sym_strides;
  auto p = lt.parent(ref);
  while (p != root) {
    const auto& l = lt.loop(p);
    auto sym = var_to_sym.at(l.var);
    auto stride = inner_sizes.at(p);
    sym_strides[sym].emplace_back(p, stride);
    p = lt.parent(p);
  }
  return sym_strides;
}

int32_t WebAssemblyCompiler::push_access_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  auto access = gen_access(node_ref, ref);
  // grab the relevant loops
  auto sym_strides = get_symbol_strides(ref, access.alloc.lca, unrolls);
  const auto& idx_expr = get_scoped_expr(access);
  // memory needs 4x for bytes sizeof(float)
  int32_t offset =
      get_unroll_offset(node_ref, ref, access.alloc.lca, idx_expr, unrolls) * 4;

  bool emitted = false;
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
      int32_t inner_stride = p.second * stride * 4;
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
  return offset;
}

void WebAssemblyCompiler::push_vector_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  // auto p = lt.parent(ref);
  // auto loop = lt.loop(p);
  // if (stack_v128.count(node_ref) &&
}

void WebAssemblyCompiler::push_float_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  if (stack_f32.count(node_ref)) {
    // it's on the stack
    return;
  } else if (local_f32.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_f32.at(node_ref);
    ASSERT(off < locals.size());
    cg->local.get(locals.at(off));
    return;
  }
  auto offset = push_access_to_stack(node_ref, ref, unrolls);
  cg->f32.load(0, memory_locations.at(resolved_reads.at(node_ref)) + offset);
}

void WebAssemblyCompiler::emit_vectorized_node(
    LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  const auto& parent = lt.parent(ref);
  if (!unrolls.count(parent)) {
    std::cerr << "Warning: cannot emit vectorized node unless unrolled";
    return emit_node(ref, unrolls);
  }
  const auto& node_ref = lt.node(ref);
  const auto off = get_unroll_offset(node_ref, ref, unrolls);
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
  if (is_reduction || node.op() == Operation::read) {
    push_float_to_stack(node_ref, ref, unrolls);
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

  if (stack_storage.count(node_ref)) {
    // pass
  } else if (local_storage.count(node_ref)) {
    const auto off = get_unroll_offset(node_ref, ref, unrolls);
    const auto& locals = local_f32.at(node_ref);
    ASSERT(off < locals.size());
    cg->local.set(locals.at(off));
  } else {
    cg->local.set(get_tmp_f32());
    auto store_offset = push_access_to_stack(node_ref, ref, unrolls);
    cg->local.get(get_tmp_f32());
    cg->f32.store(
        0, memory_locations.at(resolved_writes.at(node_ref)) + store_offset);
  }
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
      ASSERT(0) << "reset not possible for stack resident memory";
    } else if (local_f32.count(alloc.node_ref)) {
      for (const auto& local : local_f32.at(alloc.node_ref)) {
        cg->f32.const_(value(node));
        cg->local.set(local);
      }
    } else if (local_v128.count(alloc.node_ref)) {
      ASSERT(0) << "not yet supported";
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

void WebAssemblyCompiler::emit_loop(
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

  // generate any resets
  if (lt.children(lt.parent(ref)).at(0) == ref) {
    emit_reset(lt.parent(ref));
  }

  // three cases
  if (false && size == 4 && lt.children(ref).size() == 1 &&
      lt.annotation(ref) == "unroll") {  // 1. vectorized
    emit_vectorized_node(ref, unrolls);
  } else if (lt.annotation(ref) == "unroll") {  // 2. unrolled loop
    for (auto i = 0; i < size; ++i) {
      unrolls[ref] = i;
      for (auto c : lt.children(ref)) {
        emit(c, overrides, unrolls);
      }
    }
  } else {  // 3. default loop
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
    // value is now fixed
    for (auto c : lt.children(ref)) {
      emit(c, overrides, unrolls);
    }
  }
}

int32_t WebAssemblyCompiler::get_tmp_f32() const {
  if (tmp_f32 == -1) {
    tmp_f32 = cg->local(cg->f32);
  }
  return tmp_f32;
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
