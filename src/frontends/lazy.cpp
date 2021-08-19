#include "loop_tool/lazy.h"
namespace loop_tool {
namespace lazy {

std::unordered_map<size_t, CachedCompilation>& getCompilationCache() {
  static std::unordered_map<size_t, CachedCompilation> compilation_cache_;
  return compilation_cache_;
}

void TensorImpl::bind(void* data, std::vector<size_t> sizes) {
  data_ = data;
  // if no data is specified, we need to allocate
  if (data_ == nullptr) {
    owning_ = true;
  }
  if (shape_.size() == 0) {
    shape_.resize(sizes.size());
  }
  ASSERT(sizes.size() == shape_.size()) << "Invalid binding";
  ASSERT(constraints_.size() == 0) << "Already bound";
  for (auto i = 0; i < sizes.size(); ++i) {
    const auto& s = sizes.at(i);
    constraints_.emplace_back(
        std::make_pair(Expr::size(shape_.at(i)), Expr(s)));
  }
}

std::vector<void*> TensorImpl::getBuffers() const {
  if (op_ == Operation::constant) {
    return {data<void>()};
  }
  std::vector<void*> all_buffers;
  for (const auto& dep : deps_) {
    for (const auto& b : dep->getBuffers()) {
      all_buffers.emplace_back(b);
    }
  }
  return all_buffers;
}

LoopTree TensorImpl::schedule(
    IR ir,
    const std::unordered_map<int, std::pair<IR::VarRef, size_t>>& var_map)
    const {
  std::unordered_map<IR::VarRef, size_t> var_sizes;
  for (const auto& p : var_map) {
    var_sizes[p.second.first] = p.second.second;
  }
  for (const auto& n : ir.nodes()) {
    std::vector<std::pair<IR::VarRef, IR::LoopSize>> order;
    for (const auto& v : ir.all_vars(n)) {
      order.emplace_back(
          std::make_pair(v, IR::LoopSize{(int)var_sizes.at(v), 0}));
    }
    ir.set_order(n, order);
  }
  LoopTree loop_tree(ir);
  return loop_tree;
}

size_t TensorImpl::size(int dim) const {
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
  ASSERT(expr.type() == Expr::Type::value)
      << "cannot resolve symbol " << shape().at(dim).name();
  return expr.value();
}
IR::NodeRef TensorImpl::resolve(
    IR& ir,
    std::unordered_map<int, std::pair<IR::VarRef, size_t>>& var_map) const {
  std::vector<IR::NodeRef> node_deps;
  std::vector<IR::VarRef> vars;
  for (const auto& d : deps_) {
    auto node_ref = d->resolve(ir, var_map);
    node_deps.emplace_back(node_ref);
  }
  IR::NodeRef node_ref = -1;

  for (const auto& s : shape()) {
    if (!var_map.count(s.id())) {
      ASSERT(size_constraints().count(s.id()))
          << "unbound variable in compute " << s.name() << " (id: " << s.id()
          << ")";
      auto expr = size_constraints().at(s.id());
      auto size = expr.value();
      std::stringstream s_name;
      s_name << s.name();
      s_name << "_";
      s_name << s.id();
      auto var = ir.create_var(s_name.str());
      var_map[s.id()] = std::make_pair(var, size);
    }
    auto& p = var_map.at(s.id());
    vars.emplace_back(p.first);
  }
  switch (op_) {
    case Operation::name:
      ASSERT(node_deps.size() == 1) << "invalid rename (only 1 input allowed)";
      node_ref = node_deps[0];
      break;
    case Operation::view:
      ASSERT(0) << "view not yet supported";
      node_ref = node_deps[0];
      break;
    case Operation::constant:
      node_ref = ir.create_node("read", {}, vars);
      ir.add_input(node_ref);
      break;
    case Operation::add:
      node_ref = ir.create_node("add", node_deps, vars);
      break;
    case Operation::multiply:
      node_ref = ir.create_node("mul", node_deps, vars);
      break;
  }
  ASSERT(node_ref > -1) << "couldn't resolve node op: " << (int)op_;
  return node_ref;
}

void TensorImpl::collectConstraints(std::vector<Constraint>& constraints) {
  for (const auto& c : constraints_) {
    constraints.emplace_back(c);
  }
  for (const auto& d : deps_) {
    d->collectConstraints(constraints);
  }
}

void TensorImpl::propogateConstraints(
    const std::unordered_map<int, Expr>& size_constraints) {
  constraints_.clear();
  for (const auto& s : shape()) {
    auto id = s.id();
    if (size_constraints.count(id)) {
      constraints_.emplace_back(
          std::make_pair(Expr::size(s), size_constraints.at(id)));
    }
  }
  for (const auto& d : deps_) {
    d->propogateConstraints(size_constraints);
  }
}

void TensorImpl::unifyConstraints() {
  std::vector<Constraint> constraints;
  collectConstraints(constraints);
  auto new_constraints = symbolic::unify(constraints);
  std::unordered_map<int, Expr> size_constraints;
  for (const auto& c : new_constraints) {
    const auto& expr = c.first;
    if (expr.type() == Expr::Type::symbol) {
    } else {
      auto symbol = expr.args().at(0).symbol();
      size_constraints.emplace(symbol.id(), c.second);
    }
  }
  propogateConstraints(size_constraints);
}

void TensorImpl::collectSymbolMap(std::unordered_map<int, Symbol>& symbol_map) {
  // propagates all Tensor::as calls to assign symbols
  if (op_ == Operation::name) {
    ASSERT(deps_.size() == 1);
    const auto& dep_shape = deps_.at(0)->shape();
    ASSERT(dep_shape.size() == shape_.size());
    for (auto i = 0; i < shape_.size(); ++i) {
      // check there's no cycle before adding to the map
      auto s = shape_[i];
      std::unordered_set<int> seen_ids{s.id()};
      while (symbol_map.count(s.id())) {
        s = symbol_map.at(s.id());
        ASSERT(seen_ids.count(s.id()) == 0) << "unexpected cycle found in symbol map";
      }
      if (seen_ids.count(dep_shape[i].id()) == 0) {
        symbol_map[dep_shape[i].id()] = shape_[i];
      }
    }
  }
  if (op_ == Operation::view) {
  }
  for (auto d : deps_) {
    d->collectSymbolMap(symbol_map);
  }
}

void TensorImpl::propagateSymbolMap(
    const std::unordered_map<int, Symbol>& symbol_map) {
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
  for (auto d : deps_) {
    d->propagateSymbolMap(symbol_map);
  }
}

void TensorImpl::unifySymbols() {
  std::unordered_map<int, Symbol> symbol_map;
  collectSymbolMap(symbol_map);
  propagateSymbolMap(symbol_map);
}

void TensorImpl::unify() {
  unifySymbols();
  unifyConstraints();
}

void TensorImpl::populateCompilationCache() {
  unify();

  IR ir;
  std::unordered_map<int, std::pair<IR::VarRef, size_t>> var_map;
  auto node_ref = resolve(ir, var_map);

  std::vector<IR::VarRef> vars;
  size_t size = 1;
  for (const auto& s : shape()) {
    vars.emplace_back(var_map.at(s.id()).first);
    size *= var_map.at(s.id()).second;
  }
  auto out = ir.create_node("write", {node_ref}, vars);
  ir.set_outputs({out});

  auto loop_tree = schedule(ir, var_map);
  auto cc = getBackends().at("cpu")->compile(loop_tree, {}, -1);
  getCompilationCache().emplace(
      hash(), CachedCompilation{std::move(cc), ir, loop_tree, size});
}

}  // namespace lazy
}  // namespace loop_tool
