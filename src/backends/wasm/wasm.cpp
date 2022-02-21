/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/wasm.h"

using namespace loop_tool;
using namespace symbolic;

int32_t WebAssemblyCompiler::push_access_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  // grab the relevant loops
  std::unordered_map<Symbol, std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                     Hash<Symbol>>
      sym_strides;
  auto access = gen_access(node_ref, ref);
  auto p = lt.parent(ref);
  int32_t offset = 0;
  while (p != access.alloc.lca) {
    const auto& l = lt.loop(p);
    auto sym = var_to_sym.at(l.var);
    auto stride = inner_sizes.at(p);
    if (unrolls.count(p)) {
      offset += unrolls.at(p) * stride;
    } else {
      sym_strides[sym].emplace_back(p, stride);
    }
    p = lt.parent(p);
  }
  offset *= 4;

  // grab index equation
  std::vector<Expr> scoped_exprs;
  for (const auto& p : access.exprs) {
    auto expr = p.first
                    .walk([&](const Expr& e) {
                      if (e.type() == Expr::Type::symbol) {
                        if (!sym_strides.count(e.symbol())) {
                          return Expr(0);
                        }
                      }
                      return e;
                    })
                    .simplify();
    scoped_exprs.emplace_back(expr);
  }

  Expr full_expr(0);
  for (auto i = 0; i < scoped_exprs.size(); ++i) {
    auto stride = access.alloc.size(i + 1);
    const auto& expr = scoped_exprs.at(i);
    full_expr = full_expr + expr * Expr(stride);
  }
  full_expr = full_expr.simplify();

  bool emitted = false;
  for (const auto& sym : full_expr.symbols()) {
    auto stride_expr = differentiate(full_expr, sym).simplify();
    ASSERT(stride_expr.type() == Expr::Type::value) << "Invalid indexing expr";
    auto stride = stride_expr.value();
    if (stride == 0) {
      continue;
    }
    for (const auto& p : sym_strides.at(sym)) {
      int32_t inner_stride = p.second * stride;
      if (inner_stride == 0) {
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
  // float size
  if (emitted) {
    cg->i32.const_(4);
    cg->i32.mul();
  } else {
    cg->i32.const_(0);
  }
  return offset;
}

void WebAssemblyCompiler::push_vector_to_stack(IR::NodeRef node_ref,
                                               LoopTree::TreeRef ref) const {}

void WebAssemblyCompiler::push_float_to_stack(
    IR::NodeRef node_ref, LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  if (local_f32.count(node_ref)) {
    cg->local.get(local_f32.at(node_ref));
    return;
  }
  auto offset = push_access_to_stack(node_ref, ref, unrolls);
  cg->f32.load(0, memory_locations.at(resolved_reads.at(node_ref)) + offset);
}

void WebAssemblyCompiler::emit_vectorized_node(
    LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {}

void WebAssemblyCompiler::emit_node(
    LoopTree::TreeRef ref,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  const auto& node_ref = lt.node(ref);
  const auto& node = lt.ir.node(node_ref);

  if (lt.children(lt.parent(ref)).at(0) == ref) {
    emit_reset(lt.parent(ref));
  }

  int32_t store_offset = 0;
  if (local_storage.count(node_ref)) {
    ASSERT(local_f32.count(node_ref));
  } else {
    store_offset = push_access_to_stack(node_ref, ref, unrolls);
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
    case Operation::read:
    case Operation::write:
      break;
    default:
      ASSERT(0) << "Can't handle op yet for wasm " << lt.ir.dump(node_ref);
  };

  if (local_storage.count(node_ref)) {
    cg->local.set(local_f32.at(node_ref));
  } else {
    cg->f32.store(
        0, memory_locations.at(resolved_writes.at(node_ref)) + store_offset);
  }
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
    bool needs_set = lt.ir.reduction_vars(alloc.node_ref).size() &&
                     node.op() != Operation::view;
    for (const auto& input : node.inputs()) {
      if (lt.ir.node(input).op() == Operation::view &&
          !lt.scheduled.count(input)) {
        needs_set = true;
      }
    }
    if (!lt.scheduled.count(alloc.node_ref) || !needs_set) {
      continue;
    }
    if (local_f32.count(alloc.node_ref)) {
      cg->f32.const_(value(node));
      cg->local.set(local_f32.at(alloc.node_ref));
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

void WebAssemblyCompiler::emit(
    LoopTree::TreeRef ref, std::unordered_map<IR::VarRef, int> overrides,
    std::unordered_map<LoopTree::TreeRef, int32_t> unrolls) const {
  if (ref == -1) {
    auto func = cg->function({}, {}, [&]() {
      for (const auto& node_ref : local_storage) {
        local_f32[node_ref] = cg->local(cg->f32);
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
  local_f32.clear();
  local_v128.clear();
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
