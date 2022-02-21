/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/lazy.h"
namespace loop_tool {
namespace lazy {

std::unordered_map<size_t, CachedCompilation>& getCompilationCache() {
  static std::unordered_map<size_t, CachedCompilation> compilation_cache_;
  return compilation_cache_;
}

void TensorImpl::bind(void* data, std::vector<int64_t> sizes) {
  memory_.address = data;
  if (data) {
    owning_ = false;
    // we need to consider the case where we're an output
    if (deps_.size()) {
      force_recompute_ = true;
    }
  }

  // if no data is specified, we need to allocate
  if (memory_.address == nullptr) {
    owning_ = true;
  }

  if (shape_.size() == 0) {
    shape_.resize(sizes.size());
  }
  ASSERT(sizes.size() == shape_.size())
      << "Invalid binding, expected " << shape_.size() << " dims got "
      << sizes.size() << " dims";
  if (constraints_.size() > 0) {
    for (auto i = 0; i < sizes.size(); ++i) {
      for (const auto& c : constraints_) {
        if (c.first == Expr::size(shape_.at(i)) &&
            c.second.type() == Expr::Type::value) {
          ASSERT(c.second.value() == sizes.at(i))
              << "Already bound " << c.first.dump() << " to " << c.second.dump()
              << ", can't change that to " << sizes.at(i);
        }
      }
    }
  }
  for (auto i = 0; i < sizes.size(); ++i) {
    const auto& s = sizes.at(i);
    constraints_.emplace_back(
        std::make_pair(Expr::size(shape_.at(i)), Expr(s)));
  }
  updateHash();
}

std::vector<void*> TensorImpl::getInputBuffers(
    std::unordered_set<const TensorImpl*>& seen) const {
  if (seen.count(this)) {
    return {};
  }
  seen.insert(this);
  if (op_ == Operation::constant) {
    return {data<void>()};
  }
  std::vector<void*> all_buffers;
  for (const auto& dep : deps_) {
    for (const auto& b : dep->getInputBuffers(seen)) {
      all_buffers.emplace_back(b);
    }
  }
  return all_buffers;
}

LoopTree TensorImpl::schedule(
    IR& ir,
    const std::unordered_map<int, std::pair<IR::VarRef, int64_t>>& var_map)
    const {
  std::unordered_map<IR::VarRef, int64_t> var_sizes;
  for (const auto& p : var_map) {
    var_sizes[p.second.first] = p.second.second;
  }
  for (const auto& n : ir.nodes()) {
    std::vector<std::pair<IR::VarRef, IR::LoopSize>> order;
    switch (ir.node(n).op()) {
      case Operation::read:
      case Operation::view:
        // case Operation::write:
        ir.set_order(n, {});
        break;
      // case Operation::view:
      //  for (const auto& v : ir.node(n).vars()) {
      //    // for (const auto& v : ir.node(ir.node(n).inputs().at(0)).vars()) {
      //    order.emplace_back(
      //        std::make_pair(v, IR::LoopSize{(int)var_sizes.at(v), 0}));
      //  }
      //  ir.set_order(n, order);
      //  break;
      default: {
        for (const auto& v : ir.loop_vars(n)) {
          order.emplace_back(
              std::make_pair(v, IR::LoopSize{(int)var_sizes.at(v), 0}));
        }
        ir.set_order(n, order);
        break;
      }
    }
  }
  LoopTree loop_tree(ir);
  return loop_tree;
}

std::vector<int64_t> TensorImpl::sizes() const {
  if (getCompilationCache().count(hash())) {
    const auto& cc = getCompilationCache().at(hash());
    sizes_ = cc.sizes;
    cached_sizes_ = true;
  }
  if (!cached_sizes_) {
    for (auto i = 0; i < shape().size(); ++i) {
      sizes_.emplace_back(size(i));
    }
    cached_sizes_ = true;
  }
  return sizes_;
}

int64_t TensorImpl::size(int dim) const {
  const_cast<TensorImpl*>(this)->unify();
  ASSERT(dim < shape().size());
  auto id = shape().at(dim).id();
  ASSERT(size_constraints().count(id))
      << "couldn't find size of " << Expr(shape().at(dim)).dump() << "\n";
  auto expr = size_constraints().at(id);
  if (expr.type() != Expr::Type::value) {
    const_cast<TensorImpl*>(this)->unify();
    expr = size_constraints().at(id);
  }
  ASSERT(expr.can_evaluate())
      << "cannot resolve symbol " << shape().at(dim).name() << " got expr "
      << expr.dump();
  return expr.evaluate();
}

IR::NodeRef TensorImpl::resolve(
    IR& ir, std::unordered_map<int, std::pair<IR::VarRef, int64_t>>& var_map,
    std::unordered_map<const TensorImpl*, IR::NodeRef>& impl_map) const {
  if (impl_map.count(this)) {
    return impl_map.at(this);
  }
  std::vector<IR::NodeRef> node_deps;
  std::vector<IR::VarRef> vars;
  std::vector<Constraint> node_constraints;
  std::unordered_map<int, IR::VarRef> sym_var_map;

  for (const auto& d : deps_) {
    auto node_ref = d->resolve(ir, var_map, impl_map);
    node_deps.emplace_back(node_ref);
  }
  IR::NodeRef node_ref = -1;
  for (const auto& s : shape()) {
    if (!var_map.count(s.id())) {
      ASSERT(size_constraints().count(s.id()))
          << "unbound variable in compute " << s.name() << " (id: " << s.id()
          << ")";
      auto expr = size_constraints().at(s.id());
      ASSERT(expr.can_evaluate()) << "can't resolve size";
      auto size = static_cast<int64_t>(expr.evaluate());
      std::stringstream s_name;
      s_name << s.name();
      s_name << "_";
      s_name << s.id();
      auto var = ir.create_var(s_name.str());
      ASSERT(var_map.count(s.id()) == 0);
      var_map[s.id()] = std::make_pair(var, size);
    }
    auto& p = var_map.at(s.id());
    vars.emplace_back(p.first);
  }
  for (const auto& p : var_map) {
    sym_var_map[p.first] = p.second.first;
  }
  for (const auto& c : constraints_) {
    auto in_map = [&](const Expr& e) {
      for (const auto& s : e.symbols()) {
        if (!sym_var_map.count(s.id())) {
          return false;
        }
      }
      return true;
    };
    if (in_map(c.first) && in_map(c.second)) {
      node_constraints.emplace_back(c);
    }
  }

  switch (op_) {
    case Operation::name:
      ASSERT(node_deps.size() == 1) << "invalid rename (only 1 input allowed)";
      node_ref = node_deps[0];
      break;
    case Operation::view:
      node_ref = ir.create_node(Operation::view, node_deps, vars,
                                node_constraints, sym_var_map);
      break;
    case Operation::constant:
      node_ref = ir.create_node(Operation::read, {}, vars);
      ir.add_input(node_ref);
      break;
    case Operation::add:
      node_ref = ir.create_node(Operation::add, node_deps, vars);
      break;
    case Operation::subtract:
      node_ref = ir.create_node(Operation::subtract, node_deps, vars);
      break;
    case Operation::multiply:
      node_ref = ir.create_node(Operation::multiply, node_deps, vars);
      break;
    case Operation::divide:
      node_ref = ir.create_node(Operation::divide, node_deps, vars);
      break;
    case Operation::max:
      node_ref = ir.create_node(Operation::max, node_deps, vars);
      break;
    case Operation::exp:
      node_ref = ir.create_node(Operation::exp, node_deps, vars);
      break;
    case Operation::sqrt:
      node_ref = ir.create_node(Operation::sqrt, node_deps, vars);
      break;
    case Operation::reciprocal:
      node_ref = ir.create_node(Operation::reciprocal, node_deps, vars);
      break;
    case Operation::negate:
      node_ref = ir.create_node(Operation::negate, node_deps, vars);
      break;
    default:
      break;
  }
  ASSERT(node_ref > -1) << "couldn't resolve node op: " << dump(op_);
  impl_map.insert(std::make_pair(this, node_ref));
  return node_ref;
}

void TensorImpl::collectConstraints(std::vector<Constraint>& constraints,
                                    std::unordered_set<TensorImpl*>& seen) {
  for (const auto& c : constraints_) {
    constraints.emplace_back(c);
  }
  seen.insert(this);
  for (const auto& d : deps_) {
    if (seen.count(d.get())) {
      continue;
    }
    d->collectConstraints(constraints, seen);
  }
}

void TensorImpl::propagateConstraints(
    const std::vector<Constraint>& constraints,
    std::unordered_set<TensorImpl*>& seen) {
  // collect sym deps for current constraints
  // TODO: change to set
  std::vector<Symbol> symbols;
  for (const auto& c : constraints_) {
    auto collect_syms = [&](const Expr& e) {
      if (e.type() == Expr::Type::symbol) {
        symbols.emplace_back(e.symbol());
      }
      return e;
    };
    c.first.walk(collect_syms);
    c.second.walk(collect_syms);
  }
  symbols.insert(symbols.end(), shape().begin(), shape().end());
  constraints_.clear();
  for (auto& c : constraints) {
    bool insert = false;
    for (const auto& s : symbols) {
      if (c.first.contains(s) || c.second.contains(s)) {
        insert = true;
      }
    }
    if (insert) {
      constraints_.emplace_back(c);
    }
  }
  seen.insert(this);
  for (const auto& d : deps_) {
    if (seen.count(d.get())) {
      continue;
    }
    d->propagateConstraints(constraints, seen);
  }
}

void TensorImpl::unifyConstraints() {
  std::vector<Constraint> constraints;
  std::unordered_set<TensorImpl*> seen;
  collectConstraints(constraints, seen);
  auto new_constraints = symbolic::unify(constraints);
  seen.clear();
  propagateConstraints(new_constraints, seen);
}

void TensorImpl::collectSymbolMap(std::unordered_map<int, Symbol>& symbol_map,
                                  std::unordered_set<TensorImpl*>& seen) {
  // propagates all Tensor::as calls to assign symbols
  if (op_ == Operation::name) {
    ASSERT(deps_.size() == 1);
    const auto& dep_shape = deps_.at(0)->shape();
    ASSERT(dep_shape.size() == shape_.size())
        << "found shape size of " << dep_shape.size() << " dims expected "
        << shape_.size();
    for (auto i = 0; i < shape_.size(); ++i) {
      // check there's no cycle before adding to the map
      auto s = shape_[i];
      std::unordered_set<int> seen_ids{s.id()};
      while (symbol_map.count(s.id())) {
        s = symbol_map.at(s.id());
        ASSERT(seen_ids.count(s.id()) == 0)
            << "unexpected cycle found in symbol map";
      }
      if (seen_ids.count(dep_shape[i].id()) == 0) {
        symbol_map[dep_shape[i].id()] = shape_[i];
      }
    }
  }
  seen.insert(this);
  for (auto d : deps_) {
    if (seen.count(d.get())) {
      continue;
    }
    d->collectSymbolMap(symbol_map, seen);
  }
}

void TensorImpl::propagateSymbolMap(
    const std::unordered_map<int, Symbol>& symbol_map,
    std::unordered_set<TensorImpl*>& seen) {
  for (auto& s : shape_) {
    while (symbol_map.count(s.id())) {
      auto old_s = s;
      s = symbol_map.at(old_s.id());
      for (auto& constraint : constraints_) {
        constraint.first = constraint.first.replace(old_s, s);
        constraint.second = constraint.second.replace(old_s, s);
      }
    }
  }
  seen.insert(this);
  for (auto d : deps_) {
    if (seen.count(d.get())) {
      continue;
    }
    d->propagateSymbolMap(symbol_map, seen);
  }
}

void TensorImpl::unifySymbols() {
  std::unordered_map<int, Symbol> symbol_map;
  std::unordered_set<TensorImpl*> seen;
  collectSymbolMap(symbol_map, seen);
  seen.clear();
  propagateSymbolMap(symbol_map, seen);
}

void TensorImpl::unify() {
  if (unified_) {
    return;
  }
  unified_ = true;
  unifySymbols();
  unifyConstraints();
}

std::unique_ptr<Compiled> TensorImpl::backend_compile(
    const LoopTree& loop_tree) {
  // legacy
  std::unordered_set<LoopTree::TreeRef> parallel;
  loop_tree.walk([&](LoopTree::TreeRef ref, int) {
    if (loop_tree.annotation(ref).find("parallel") != std::string::npos) {
      parallel.insert(ref);
    }
  });
  return getDefaultBackend()->compile(loop_tree, parallel, -1);
}

void TensorImpl::populateCompilationCache() {
  IR ir;
  std::unordered_map<int, std::pair<IR::VarRef, int64_t>> var_map;
  std::tie(ir, var_map) = lower();
  auto loop_tree = schedule(ir, var_map);
  int64_t size = 1;
  std::vector<int64_t> sizes;
  for (const auto& s : shape()) {
    auto size_ = var_map.at(s.id()).second;
    size *= size_;
    sizes.emplace_back(size_);
  }
  auto cc = backend_compile(loop_tree);
  getCompilationCache().emplace(
      hash(), CachedCompilation{std::move(cc), ir, loop_tree, size, sizes});
}

}  // namespace lazy
}  // namespace loop_tool
