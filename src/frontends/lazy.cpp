#include "loop_tool/lazy.h"
namespace loop_tool {
namespace lazy {

std::unordered_map<size_t, CachedCompilation>& getCompilationCache() {
  static std::unordered_map<size_t, CachedCompilation> compilation_cache_;
  return compilation_cache_;
}

void TensorImpl::bind(void* data, std::vector<size_t> sizes) {
  //if (data) {
  //  std::cerr << "WARNING deprecated feature TensorImpl::bind\n";
  //}
  memory_.address = data;
  // if no data is specified, we need to allocate
  if (memory_.address == nullptr) {
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
    IR& ir,
    const std::unordered_map<int, std::pair<IR::VarRef, size_t>>& var_map)
    const {
  std::unordered_map<IR::VarRef, size_t> var_sizes;
  for (const auto& p : var_map) {
    var_sizes[p.second.first] = p.second.second;
  }
  for (const auto& n : ir.nodes()) {
    std::vector<std::pair<IR::VarRef, IR::LoopSize>> order;
    for (const auto& v : ir.loop_vars(n)) {
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
  std::unordered_map<int, IR::VarRef> sym_var_map;
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
  for (auto& p : var_map) {
    sym_var_map[p.first] = p.second.first;
  }

  switch (op_) {
    case Operation::name:
      ASSERT(node_deps.size() == 1) << "invalid rename (only 1 input allowed)";
      node_ref = node_deps[0];
      break;
    case Operation::view:
      node_ref = ir.create_node(Operation::view, node_deps, vars, constraints_,
                                sym_var_map);
      break;
    case Operation::constant:
      node_ref = ir.create_node(Operation::read, {}, vars);
      ir.add_input(node_ref);
      break;
    case Operation::add:
      node_ref = ir.create_node(Operation::add, node_deps, vars);
      break;
    case Operation::multiply:
      node_ref = ir.create_node(Operation::multiply, node_deps, vars);
      break;
    default:
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

void TensorImpl::propagateConstraints(
    const std::vector<Constraint>& constraints) {
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
  for (const auto& d : deps_) {
    d->propagateConstraints(constraints);
  }
}

void TensorImpl::unifyConstraints() {
  std::vector<Constraint> constraints;
  collectConstraints(constraints);
  auto new_constraints = symbolic::unify(constraints);
  propagateConstraints(new_constraints);
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
        ASSERT(seen_ids.count(s.id()) == 0)
            << "unexpected cycle found in symbol map";
      }
      if (seen_ids.count(dep_shape[i].id()) == 0) {
        symbol_map[dep_shape[i].id()] = shape_[i];
      }
    }
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
  if (unified_) {
    return;
  }
  unified_ = true;
  unifySymbols();
  unifyConstraints();
}

std::unique_ptr<Compiled> TensorImpl::backend_compile(
    const LoopTree& loop_tree) {
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
  std::unordered_map<int, std::pair<IR::VarRef, size_t>> var_map;
  std::tie(ir, var_map) = lower();
  auto loop_tree = schedule(ir, var_map);
  size_t size = 1;
  for (const auto& s : shape()) {
    size *= var_map.at(s.id()).second;
  }
  auto cc = backend_compile(loop_tree);
  getCompilationCache().emplace(
      hash(), CachedCompilation{std::move(cc), ir, loop_tree, size});
}

}  // namespace lazy
}  // namespace loop_tool
