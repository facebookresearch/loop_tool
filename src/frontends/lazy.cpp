#include "loop_tool/lazy.h"
namespace loop_tool {
namespace lazy {

const int getNewSymbolId() {
  static int symbol_count_ = 0;
  return symbol_count_++;
}

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
    constraints_.emplace(shape_.at(i).id(), Expr(s));
  }
}

void TensorImpl::unify(std::unordered_map<int, Symbol> symbol_map) {
  for (auto& s : shape_) {
    while (symbol_map.count(s.id())) {
      auto old_id = s.id();
      s = symbol_map.at(old_id);
      if (constraints_.count(old_id)) {
        constraints_.emplace(s.id(), constraints_.at(old_id));
        constraints_.erase(old_id);
      }
    }
  }
  if (op_ == Operation::view) {
    ASSERT(deps_.size() == 1);
    const auto& dep_shape = deps_.at(0)->shape();
    for (auto i = 0; i < shape_.size(); ++i) {
      symbol_map[dep_shape[i].id()] = shape_[i];
    }
  }
  for (auto d : deps_) {
    d->unify(symbol_map);
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
      ASSERT(constraints_.count(s.id()))
          << "unbound variable in compute " << s.name();
      auto expr = constraints_.at(s.id());
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
    case Operation::view:
      ASSERT(node_deps.size() == 1) << "invalid view";
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
  return node_ref;
}

void TensorImpl::populateCompilationCache() {
  // collect constraints
  // unify constraints
  // propagate updated constraints
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

// TODO: AC unification algorithm i.e. Expr = Expr constraints with
// associativity/commutativity
std::vector<std::pair<Symbol, Expr>> unify(
    std::vector<std::pair<Symbol, Expr>> constraints) {
  std::function<Expr(Expr)> eval_expr;
  // Symbol.id() -> value
  std::unordered_map<int, Expr> replacements;
  auto pass = [&]() -> bool {
    bool updated = false;
    for (auto& p : constraints) {
      p.second = eval_expr(p.second);
      if (!replacements.count(p.first.id())) {
        replacements.emplace(p.first.id(), p.second);
        updated = true;
      }
      if (replacements.at(p.first.id()) != p.second) {
        replacements.at(p.first.id()) = p.second;
        updated = true;
      }
    }
    return updated;
  };

  eval_expr = [&](Expr e) -> Expr {
    if (e.type() == Expr::Type::value) {
      return e;
    } else if (e.type() == Expr::Type::symbol) {
      auto id = e.symbol().id();
      if (replacements.count(id)) {
        return replacements.at(id);
      }
      return e;
    }
    ASSERT(e.type() == Expr::Type::function);
    ASSERT(e.args().size() == 2);
    auto lhs = eval_expr(e.args().at(0));
    auto rhs = eval_expr(e.args().at(1));
    if (e.op() == Operation::add) {
      if (lhs.type() == Expr::Type::value && rhs.type() == Expr::Type::value) {
        return Expr(lhs.value() + rhs.value());
      }
      return lhs + rhs;
    } else if (e.op() == Operation::multiply) {
      if (lhs.type() == Expr::Type::value && rhs.type() == Expr::Type::value) {
        return Expr(lhs.value() * rhs.value());
      }
      return lhs * rhs;
    }
    ASSERT(0) << "unknown expression op";
    return e;
  };

  while (pass())
    ;

  std::vector<std::pair<Symbol, Expr>> out;
  for (const auto& p : constraints) {
    out.emplace_back(std::make_pair(p.first, eval_expr(p.second)));
  }
  return out;
}

}  // namespace lazy
}  // namespace loop_tool
