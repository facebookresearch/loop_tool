/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/wasm.h"

using namespace loop_tool;
using namespace symbolic;

void WebAssemblyCompiler::push_access_to_stack(IR::NodeRef node_ref,
                                               LoopTree::TreeRef ref) const {
  // grab the relevant loops
  std::unordered_map<Symbol, std::vector<std::pair<LoopTree::TreeRef, int64_t>>,
                     Hash<Symbol>>
      sym_strides;
  auto access = gen_access(node_ref, ref);
  auto p = lt.parent(ref);
  while (p != access.alloc.lca) {
    const auto& l = lt.loop(p);
    auto sym = var_to_sym.at(l.var);
    auto stride = inner_sizes.at(p);
    sym_strides[sym].emplace_back(p, stride);
    p = lt.parent(p);
  }

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
}

void WebAssemblyCompiler::push_vector_to_stack(IR::NodeRef node_ref,
                                               LoopTree::TreeRef ref) const {
  const auto& vars = lt.ir.node(node_ref).vars();
  const auto& var_set = to_set(vars);
  // three cases
  auto dim = lt.loop(lt.parent(ref)).var;
  if (vars.back() == dim) {  // 1. contiguous load
    push_access_to_stack(node_ref, ref);
    cg->v128.load(0, memory_locations.at(node_ref));
  } else if (var_set.count(dim)) {  // 2. strided load
    ASSERT(0) << "Gather not yet supported";
  } else {  // 3. broadcast load
    push_access_to_stack(node_ref, ref);
    cg->v128.load32_splat(0, memory_locations.at(node_ref));
  }
}

void WebAssemblyCompiler::push_float_to_stack(IR::NodeRef node_ref,
                                              LoopTree::TreeRef ref) const {
  if (local_f32.count(node_ref)) {
    cg->local.get(local_f32.at(node_ref));
    return;
  }
  push_access_to_stack(node_ref, ref);
  cg->f32.load(0, memory_locations.at(node_ref));
  // maybe save to local?
}

void WebAssemblyCompiler::emit_vectorized_node(LoopTree::TreeRef ref) const {}

void WebAssemblyCompiler::emit_node(LoopTree::TreeRef ref) const {
  const auto& node_ref = lt.node(ref);
  const auto& node = lt.ir.node(node_ref);

  if (lt.children(lt.parent(ref)).at(0) == ref) {
    emit_reset(lt.parent(ref));
  }

  push_access_to_stack(node_ref, ref);

  for (const auto& inp_ref : node.inputs()) {
    push_access_to_stack(inp_ref, ref);
    cg->f32.load(0, memory_locations.at(resolved_reads.at(inp_ref)));
  }
  bool is_reduction = lt.ir.reduction_vars(node_ref).size();
  if (is_reduction) {
    push_access_to_stack(node_ref, ref);
    cg->f32.load(0, memory_locations.at(node_ref));
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

  cg->f32.store(0, memory_locations.at(node_ref));
}

void WebAssemblyCompiler::emit_reset(LoopTree::TreeRef ref) const {
  auto value = [&](const Node& node) -> float {
    if (node.op() == Operation::add) {
      return 0;
    } else if (node.op() == Operation::multiply) {
      return 1;
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

    auto iter = cg->local(cg->i32);
    cg->i32.const_(0);
    cg->local.set(iter);
    cg->loop(cg->void_);

    cg->local.get(iter);
    cg->f32.const_(value(node));
    cg->f32.store(0, memory_locations.at(alloc.node_ref));

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

void WebAssemblyCompiler::emit_loop(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
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
    emit_vectorized_node(ref);
  } else if (lt.annotation(ref) == "unroll") {  // 2. unrolled loop
    // generate any loop header logic
    auto iter = cg->local(cg->i32);
    iterators[ref] = iter;

    for (auto i = 0; i < size; ++i) {
      cg->i32.const_(i);
      cg->local.set(iter);
      // generate body code
      for (auto c : lt.children(ref)) {
        emit(c, overrides);
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
      emit(c, overrides);
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
      emit(c, overrides);
    }
  }
}

void WebAssemblyCompiler::emit(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  if (ref == -1) {
    auto func = cg->function({}, {}, [&]() {
      for (auto c : lt.roots) {
        emit(c, overrides);
      }
    });
    cg->export_(func, "fn");
  } else if (lt.kind(ref) == LoopTree::NODE) {
    emit_node(ref);
  } else {
    emit_loop(ref, overrides);
  }
}

std::vector<uint8_t> WebAssemblyCompiler::emit() const {
  cg = std::move(std::make_shared<wasmblr::CodeGenerator>());
  local_f32.clear();
  local_v128.clear();
  memory_locations.clear();
  iterators.clear();

  // we can determine exactly which memory can reside in float32, vec4 and
  // multiples of them usage based:
  // 1. in vectorized node, used as fp32 or used as full vec
  // for m:4 unroll
  //   for n:4 unroll // this is a vectorized node, we know c[m,n] is going to
  //   be vec
  //     c[m,n] = a[m] * b[n]
  // for m:4 unroll
  //   for n:4 unroll
  //     d[m,n] += c[m,n]

  // for n:4 unroll
  //   for m:4 unroll
  //     c[m,n] = a[m] * b[n] // a is vectorized, but we get a bunch of c
  // for m:4 unroll
  //   for n:4 unroll // now c can be vectorized, but we find it isn't
  //     d[m,n] += c[m,n]

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
  emit(-1, {});

  return cg->emit();
}
