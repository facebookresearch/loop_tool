#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "loop_tool/backend.h"
#include "loop_tool/ir.h"

namespace detail {

template <typename... Conds>
struct and_ : std::true_type {};

template <typename Cond, typename... Conds>
struct and_<Cond, Conds...>
    : std::conditional<Cond::value, and_<Conds...>, std::false_type>::type {};
template <typename Target, typename... Ts>
using areT = and_<std::is_same<Ts, Target>...>;

}  // namespace detail

namespace loop_tool {
namespace lazy {

const int getNewSymbolId();

struct Symbol {
  // TODO replace with smaller construct
  std::string name_;
  int id_ = -1;
  Symbol() : id_(getNewSymbolId()) {
    std::stringstream ss;
    ss << "__" << id_;
    name_ = ss.str();
  }
  Symbol(std::string name) : id_(getNewSymbolId()), name_(name) {}
  Symbol(const Symbol& s) : id_(s.id_), name_(s.name_) {}
  const int id() const { return id_; }
  bool operator==(const Symbol& s) const { return s.id() == id_; }
  const std::string& name() const { return name_; }
};

struct Expr {
  size_t val_;
  explicit Expr(size_t val) : val_(val){};
  Expr() = delete;
  operator size_t() const { return val_; }
};

// wrapped by shared ptr for convenience
struct TensorImpl {
  Operation op_ = Operation::constant;
  mutable void* data_ = nullptr;
  mutable bool owning_ = false;
  std::vector<Symbol> shape_;
  std::unordered_map<int, Expr> constraints_;
  std::vector<std::shared_ptr<TensorImpl>> deps_;

  ~TensorImpl() {
    if (owning_) {
      free(data_);
    }
  }

  TensorImpl(void* data, std::vector<size_t> sizes) : op_(Operation::constant) {
    bind(data, sizes);
  }
  void bind(void* data, std::vector<size_t> sizes) {
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

  TensorImpl(std::vector<Symbol> shape)
      : op_(Operation::constant), shape_(shape) {}
  TensorImpl(Operation op, std::vector<Symbol> shape,
             std::vector<std::shared_ptr<TensorImpl>> deps)
      : op_(op), shape_(shape), deps_(deps) {}

  inline const std::vector<Symbol>& shape() const { return shape_; }

  void unify(std::unordered_map<int, Symbol> symbol_map = {}) {
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

  template <typename T>
  T* data() const {
    if (owning_ && !data_) {
      size_t size = 1;
      for (auto i = 0; i < shape().size(); ++i) {
        ASSERT(constraints_.count(shape()[i].id()))
            << "cannot allocate owned tensor, size for " << shape()[i].name()
            << "not provided";
        size *= constraints_.at(shape()[i].id());
      }
      data_ = malloc(sizeof(float) * size);
    }
    if (data_) {
      return static_cast<T*>(data_);
    }
    const_cast<TensorImpl*>(this)->unify();
    IR ir;
    std::unordered_map<int, std::pair<IR::VarRef, size_t>> var_map;
    std::unordered_map<IR::NodeRef, void*> data_map;
    auto node_ref = resolve(ir, var_map, data_map);
    std::vector<IR::VarRef> vars;
    size_t size = 1;
    for (const auto& s : shape()) {
      vars.emplace_back(var_map.at(s.id()).first);
      size *= var_map.at(s.id()).second;
    }
    auto out = ir.create_node("write", {node_ref}, vars);
    ir.set_outputs({out});
    data_ = malloc(sizeof(float) * size);
    owning_ = true;
    data_map[out] = data_;

    auto loop_tree = schedule(ir, var_map);
    std::vector<void*> buffers;
    for (const auto& n : ir.inputs()) {
      buffers.emplace_back(data_map.at(n));
    }
    for (const auto& n : ir.outputs()) {
      buffers.emplace_back(data_map.at(n));
    }
    auto cc = getBackends().at("cpu")->compile(loop_tree, {}, -1);
    cc->run(buffers, true);
    return static_cast<T*>(data_);
  };

  LoopTree schedule(
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

  IR::NodeRef resolve(
      IR& ir, std::unordered_map<int, std::pair<IR::VarRef, size_t>>& var_map,
      std::unordered_map<IR::NodeRef, void*>& data_map) const {
    std::vector<IR::NodeRef> node_deps;
    std::vector<IR::VarRef> vars;
    for (const auto& d : deps_) {
      auto node_ref = d->resolve(ir, var_map, data_map);
      node_deps.emplace_back(node_ref);
    }
    IR::NodeRef node_ref = -1;

    for (const auto& s : shape()) {
      if (!var_map.count(s.id())) {
        ASSERT(constraints_.count(s.id()))
            << "unbound variable in compute " << s.name();
        auto expr = constraints_.at(s.id());
        auto size = (size_t)expr;
        auto var = ir.create_var(s.name());
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
        data_map[node_ref] = data_;
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
};

struct Tensor {
  std::shared_ptr<TensorImpl> impl_;
  Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
  Tensor(std::vector<Symbol> shape)
      : impl_(std::make_shared<TensorImpl>(shape)) {}

  template <typename... Symbols,
            std::enable_if_t<detail::areT<Symbol, Symbols...>::value, int> = 0>
  Tensor(Symbols... symbols)
      : impl_(std::make_shared<TensorImpl>(std::vector<Symbol>{symbols...})) {}

  template <typename... Sizes,
            std::enable_if_t<detail::areT<int, Sizes...>::value, int> = 0>
  Tensor(Sizes... sizes)
      : impl_(std::make_shared<TensorImpl>(nullptr,
                                           std::vector<size_t>{sizes...})) {}

  Tensor(void* data, std::vector<size_t> sizes)
      : impl_(std::make_shared<TensorImpl>(data, sizes)) {}
  void bind(void* data, std::vector<size_t> sizes) {
    impl()->bind(data, sizes);
  }
  Tensor operator*(const Tensor& rhs) const {
    std::unordered_set<int> ids;
    std::vector<Symbol> new_shape;
    for (auto& symbol : impl_->shape()) {
      ids.insert(symbol.id());
      new_shape.emplace_back(symbol);
    }
    for (auto& symbol : rhs.shape()) {
      if (ids.count(symbol.id())) {
        continue;
      }
      new_shape.emplace_back(symbol);
    }
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_, rhs.impl()};
    auto new_impl =
        std::make_shared<TensorImpl>(Operation::multiply, new_shape, deps);
    return Tensor(new_impl);
  }

  template <typename... Args>
  Tensor sum(const Args&... args) {
    std::unordered_set<int> reduction = {args.id()...};
    std::vector<Symbol> new_shape;
    for (const auto& s : shape()) {
      if (reduction.count(s.id())) {
        continue;
      }
      new_shape.emplace_back(s);
    }
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    return Tensor(
        std::make_shared<TensorImpl>(Operation::add, new_shape, deps));
  }

  template <typename... Args>
  Tensor as(const Args&... args) {
    std::vector<Symbol> shape{args...};
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    return Tensor(std::make_shared<TensorImpl>(Operation::view, shape, deps));
  }

  std::shared_ptr<TensorImpl> impl() const { return impl_; }
  std::vector<Symbol> shape() const { return impl_->shape(); }

  template <typename T>
  T* data() const {
    return impl()->template data<T>();
  };
};

}  // namespace lazy
}  // namespace loop_tool
