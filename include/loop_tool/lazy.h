/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "loop_tool/backend.h"
#include "loop_tool/compile.h"
#include "loop_tool/ir.h"
#include "loop_tool/symbolic.h"

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

using Symbol = loop_tool::symbolic::Symbol;
using Expr = loop_tool::symbolic::Expr;
using Constraint = loop_tool::symbolic::Constraint;

template <typename... Args>
std::vector<Expr> Index(const Args&... args) {
  return std::vector<Expr>{Expr(args)...};
}

struct CachedCompilation {
  std::shared_ptr<Compiled> compilation;
  IR ir;
  LoopTree loop_tree;
  int64_t output_size;
  std::vector<int64_t> sizes;
};

std::unordered_map<size_t, CachedCompilation>& getCompilationCache();

// wrapped by shared ptr for convenience
struct TensorImpl {
  size_t hash_ = 0;
  bool unified_ = false;
  Operation op_ = Operation::constant;
  mutable Memory memory_;
  mutable bool owning_ = true;
  mutable bool force_recompute_ = false;
  std::vector<Symbol> shape_;
  std::vector<Constraint> constraints_;
  std::vector<std::shared_ptr<TensorImpl>> deps_;

  std::unordered_map<int, Expr> size_constraints() const {
    std::unordered_map<int, Expr> out;
    for (auto c : constraints_) {
      auto expr = c.first;
      bool size_expr = (expr.op() == symbolic::Op::size) &&
                       (expr.args().size() == 1) &&
                       (expr.args().at(0).type() == Expr::Type::symbol);
      if (size_expr) {
        out.emplace(expr.args().at(0).symbol().id(), c.second);
      }
    }
    return out;
  }

  std::vector<Constraint> constraints() const { return constraints_; }

  void updateHash() {
    auto h = symbolic::hash((size_t)op_);
    h = symbolic::hash(h ^ shape_.size());
    auto cm = constraints();
    for (auto& p : cm) {
      h = symbolic::hash(p.first.hash());
      h = symbolic::hash(p.second.hash());
    }
    for (const auto& d : deps_) {
      h = symbolic::hash(h ^ d->hash());
    }
    hash_ = h;
  }

  inline size_t hash() const { return hash_; }
  inline bool owning() const { return owning_; }

  ~TensorImpl() {
    if (owning_) {
      getDefaultHardware()->free(memory_);
    }
  }

  TensorImpl(void* data, std::vector<int64_t> sizes)
      : op_(Operation::constant) {
    bind(data, sizes);
    updateHash();
  }
  void bind(void* data, std::vector<int64_t> sizes);

  TensorImpl(std::vector<Symbol> shape)
      : op_(Operation::constant), shape_(shape) {
    updateHash();
  }
  TensorImpl(Operation op, std::vector<Symbol> shape,
             std::vector<std::shared_ptr<TensorImpl>> deps,
             std::vector<Constraint> constraints = {})
      : op_(op), shape_(shape), deps_(deps), constraints_(constraints) {
    updateHash();

    if (op != Operation::view) {
      return;
    }

    // collect all the known symbols
    std::unordered_map<int, Symbol> sym_lhs;
    for (auto& p : constraints) {
      p.first.walk([&](const Expr& e) {
        if (e.type() == Expr::Type::symbol) {
          sym_lhs[e.symbol().id()] = e.symbol();
        }
        return e;
      });
    }
    // assume all input shapes are calculated
    for (auto& dep : deps_) {
      for (auto& sym : dep->shape()) {
        sym_lhs[sym.id()] = sym;
      }
    }
    for (auto& sym : shape_) {
      if (sym_lhs.count(sym.id())) {
        continue;
      }
      for (auto& c : constraints) {
        c.second.walk([&](const Expr& e) {
          if (e.type() == Expr::Type::symbol) {
            if (sym_lhs.count(e.symbol().id()) == 0) {
              auto isolated_constraint = isolate(c, e.symbol());
              constraints_.emplace_back(isolated_constraint);
            }
          }
          return e;
        });
      }
    }
  }

  inline const std::vector<Symbol>& shape() const { return shape_; }
  int64_t size(int dim) const;
  std::vector<int64_t> sizes() const;

  // for these methods, int is Symbol::id
  void collectSymbolMap(std::unordered_map<int, Symbol>& symbol_map);
  void propagateSymbolMap(const std::unordered_map<int, Symbol>& symbol_map);
  void unifySymbols();
  void collectConstraints(std::vector<Constraint>& constraints);
  void propagateConstraints(const std::vector<Constraint>& constraints);
  void unifyConstraints();
  void unify();
  void populateCompilationCache();
  std::unique_ptr<Compiled> backend_compile(const LoopTree& lt);

  std::vector<void*> getInputBuffers(
      std::unordered_set<const TensorImpl*>& seen) const;

  template <typename T>
  T* data() const {
    // data cache was hit
    if (memory_.address && !force_recompute_) {
      return static_cast<T*>(memory_.address);
    }
    force_recompute_ = false;

    auto alloc = [&](size_t size) { return getDefaultHardware()->alloc(size); };

    // we're just a buffer
    if (owning_ && !memory_.address && deps_.size() == 0) {
      size_t size = 1;
      for (auto i = 0; i < shape().size(); ++i) {
        ASSERT(size_constraints().count(shape()[i].id()))
            << "cannot allocate owned tensor, size for " << shape()[i].name()
            << "not provided";
        size *= size_constraints().at(shape()[i].id()).value();
      }
      memory_ = alloc(sizeof(float) * size);
      return static_cast<T*>(memory_.address);
    }

    // compile
    if (!getCompilationCache().count(hash())) {
      const_cast<TensorImpl*>(this)->populateCompilationCache();
    }
    auto& cc = getCompilationCache().at(hash());

    if (owning_) {
      memory_ = alloc(sizeof(float) * cc.output_size);
    } else {
      ASSERT(memory_.address);
    }

    std::unordered_set<const TensorImpl*> seen;
    auto buffers = getInputBuffers(seen);
    buffers.emplace_back(memory_.address);
    cc.compilation->run(buffers, true);
    return static_cast<T*>(memory_.address);
  };

  std::pair<IR, std::unordered_map<int, std::pair<IR::VarRef, int64_t>>> lower()
      const {
    const_cast<TensorImpl*>(this)->unify();
    IR ir;
    std::unordered_map<int, std::pair<IR::VarRef, int64_t>> var_map;
    std::unordered_map<const TensorImpl*, IR::NodeRef> impl_map;
    auto node_ref = resolve(ir, var_map, impl_map);

    std::vector<IR::VarRef> vars;
    for (const auto& s : shape()) {
      vars.emplace_back(var_map.at(s.id()).first);
    }
    auto out = ir.create_node(Operation::write, {node_ref}, vars);
    ir.set_outputs({out});
    return std::make_pair(ir, var_map);
  }

  inline IR ir() const {
    auto h = hash();
    if (getCompilationCache().count(h)) {
      auto& cc = getCompilationCache().at(h);
      return cc.ir;
    }
    auto ll = lower();
    return schedule(ll.first, ll.second).ir;
  }

  inline LoopTree loop_tree() const {
    auto h = hash();
    if (getCompilationCache().count(h)) {
      auto& cc = getCompilationCache().at(h);
      return cc.loop_tree;
    }
    auto ll = lower();
    return schedule(ll.first, ll.second);
  }

  inline void compile() {
    if (!getCompilationCache().count(hash())) {
      populateCompilationCache();
    }
  }

  inline std::string code() const {
    auto compiler = Compiler(loop_tree());
    return compiler.gen_string();
  }

  inline void set(const IR& ir) {
    auto h = hash();

    if (!getCompilationCache().count(h)) {
      compile();
    }
    auto& cc = getCompilationCache().at(h);
    LoopTree loop_tree(ir);
    auto new_cc = backend_compile(loop_tree);
    cc = CachedCompilation{std::move(new_cc), ir, loop_tree, cc.output_size,
                           cc.sizes};
  }

  inline void set(const LoopTree& loop_tree) {
    auto h = hash();
    if (!getCompilationCache().count(h)) {
      compile();
    }
    auto& cc = getCompilationCache().at(h);
    auto new_cc = backend_compile(loop_tree);
    cc = CachedCompilation{std::move(new_cc), loop_tree.ir, loop_tree,
                           cc.output_size, cc.sizes};
  }

  inline std::shared_ptr<Compiled> compiled() {
    auto h = hash();
    if (!getCompilationCache().count(h)) {
      compile();
    }
    auto& cc = getCompilationCache().at(h);
    return cc.compilation;
  }

  LoopTree schedule(
      IR& ir,
      const std::unordered_map<int, std::pair<IR::VarRef, int64_t>>& var_map)
      const;
  IR::NodeRef resolve(
      IR& ir, std::unordered_map<int, std::pair<IR::VarRef, int64_t>>& var_map,
      std::unordered_map<const TensorImpl*, IR::NodeRef>& impl_map) const;
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
      : impl_(std::make_shared<TensorImpl>(
            nullptr, std::vector<int64_t>{static_cast<int64_t>(sizes)...})) {}

  template <typename... Sizes,
            std::enable_if_t<detail::areT<int64_t, Sizes...>::value, int> = 0>
  Tensor(Sizes... sizes)
      : impl_(std::make_shared<TensorImpl>(nullptr,
                                           std::vector<int64_t>{sizes...})) {}

  Tensor(void* data, std::vector<int64_t> sizes)
      : impl_(std::make_shared<TensorImpl>(data, sizes)) {}
  Tensor(std::vector<int64_t> sizes)
      : impl_(std::make_shared<TensorImpl>(nullptr, sizes)) {}
  void bind(void* data, std::vector<int64_t> sizes) {
    impl()->bind(data, sizes);
  }

  inline std::vector<Symbol> broadcast_shape(const Tensor& rhs) const {
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
    return new_shape;
  }

  Tensor operator*(const Tensor& rhs) const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_, rhs.impl()};
    auto new_impl = std::make_shared<TensorImpl>(Operation::multiply,
                                                 broadcast_shape(rhs), deps);
    return Tensor(new_impl);
  }

  Tensor operator/(const Tensor& rhs) const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_, rhs.impl()};
    auto new_impl = std::make_shared<TensorImpl>(Operation::divide,
                                                 broadcast_shape(rhs), deps);
    return Tensor(new_impl);
  }

  Tensor operator+(const Tensor& rhs) const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_, rhs.impl()};
    auto new_impl = std::make_shared<TensorImpl>(Operation::add,
                                                 broadcast_shape(rhs), deps);
    return Tensor(new_impl);
  }

  Tensor operator-(const Tensor& rhs) const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_, rhs.impl()};
    auto new_impl = std::make_shared<TensorImpl>(Operation::subtract,
                                                 broadcast_shape(rhs), deps);
    return Tensor(new_impl);
  }

  Tensor max(const Tensor& rhs) const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_, rhs.impl()};
    auto new_impl = std::make_shared<TensorImpl>(Operation::max,
                                                 broadcast_shape(rhs), deps);
    return Tensor(new_impl);
  }

  Tensor operator|(const Tensor& rhs) const {
    ASSERT(impl_->shape().size() == rhs.impl()->shape().size());
    std::vector<Symbol> out_shape;
    std::vector<Constraint> constraints;
    for (auto i = 0; i < impl_->shape().size(); ++i) {
      auto lhs_sym = impl_->shape().at(i);
      auto rhs_sym = rhs.impl()->shape().at(i);
      if (lhs_sym == rhs_sym) {
        out_shape.emplace_back(lhs_sym);
      } else {
        auto new_sym = Symbol(lhs_sym.name() + rhs_sym.name());
        out_shape.emplace_back(new_sym);
        constraints.emplace_back(Constraint(new_sym, lhs_sym));
        constraints.emplace_back(
            Constraint(new_sym, rhs_sym + Expr::size(lhs_sym)));
        constraints.emplace_back(Constraint(
            Expr::size(new_sym), Expr::size(lhs_sym) + Expr::size(rhs_sym)));
      }
    }
    return this->to(out_shape, constraints) + rhs.to(out_shape, constraints);
  }

  Tensor pad(Symbol padded_dim, int64_t pre, int64_t post) {
    ASSERT(pre >= 0) << "cannot pad by a negative number";
    ASSERT(post >= 0) << "cannot pad by a negative number";
    if (pre == 0 && post == 0) {
      return *this;
    }
    auto new_sym = Symbol(padded_dim.name() + "_p_" + std::to_string(pre) + "_" + std::to_string(post));
    std::vector<Symbol> out_shape;
    for (const auto& sym : shape()) {
      if (sym == padded_dim) {
        out_shape.emplace_back(new_sym);
      } else {
        out_shape.emplace_back(sym);
      }
    }
    return this->to(out_shape, Constraint(new_sym, padded_dim + Expr(pre)),
                    Constraint(Expr::size(new_sym),
                               Expr::size(padded_dim) + Expr(post + pre)));
  }

  Tensor pad(Symbol padded_dim, int64_t amount) {
    return pad(padded_dim, amount, amount);
  }

  Tensor transpose(std::vector<int> order) {
    ASSERT(order.size() == shape().size()) << "invalid transpose";
    std::vector<Symbol> new_shape;
    for (auto idx : order) {
      ASSERT(idx < order.size()) << "invalid transpose";
      new_shape.emplace_back(shape().at(idx));
    }
    auto new_impl = std::make_shared<TensorImpl>(*impl_);
    new_impl->shape_ = new_shape;
    return Tensor(new_impl);
  }

  Tensor transpose(std::vector<Symbol> new_shape) {
    ASSERT(new_shape.size() == shape().size()) << "invalid transpose";
    auto new_impl = std::make_shared<TensorImpl>(*impl_);
    new_impl->shape_ = new_shape;
    return Tensor(new_impl);
  }

  Tensor sum(std::vector<Symbol> reduction_vars) const {
    std::unordered_set<int> reduction;
    for (auto rv : reduction_vars) {
      reduction.insert(rv.id());
    }
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
  Tensor sum(const Args&... args) const {
    std::vector<Symbol> reduction_vars = {args...};
    return sum(reduction_vars);
  }

  Tensor exp() const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    auto new_impl =
        std::make_shared<TensorImpl>(Operation::exp, impl_->shape(), deps);
    return Tensor(new_impl);
  }

  Tensor sqrt() const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    auto new_impl =
        std::make_shared<TensorImpl>(Operation::sqrt, impl_->shape(), deps);
    return Tensor(new_impl);
  }

  Tensor operator-() const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    auto new_impl =
        std::make_shared<TensorImpl>(Operation::negate, impl_->shape(), deps);
    return Tensor(new_impl);
  }

  Tensor reciprocal() const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    auto new_impl = std::make_shared<TensorImpl>(Operation::reciprocal,
                                                 impl_->shape(), deps);
    return Tensor(new_impl);
  }

  inline Tensor as(std::vector<Symbol> shape) {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    return Tensor(std::make_shared<TensorImpl>(Operation::name, shape, deps));
  }

  template <typename... Args>
  Tensor as(const Args&... args) {
    std::vector<Symbol> shape{args...};
    return as(shape);
  }

  inline Tensor to(std::vector<Symbol> shape,
                   std::vector<Constraint> constraints) const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};

    return Tensor(std::make_shared<TensorImpl>(Operation::view, shape, deps,
                                               constraints));
  }

  template <typename... Constraints>
  Tensor to(std::vector<Symbol> shape, const Constraints&... args) const {
    std::vector<Constraint> constraints{args...};
    return to(shape, constraints);
  }

  inline std::shared_ptr<TensorImpl> impl() const { return impl_; }
  inline bool owning() const { return impl_->owning(); }
  inline std::vector<Symbol> shape() const { return impl_->shape(); }
  inline size_t size(int dim) const { return impl_->size(dim); }
  inline std::vector<int64_t> sizes() const { return impl_->sizes(); }
  inline size_t numel() const {
    size_t total = 1;
    for (auto i = 0; i < shape().size(); ++i) {
      total *= size(i);
    }
    return total;
  }
  inline LoopTree loop_tree() const { return impl_->loop_tree(); }
  inline IR ir() const { return impl_->ir(); }
  inline std::string code() const { return impl_->code(); }
  inline void set(const IR& ir) { impl_->set(ir); }
  inline void set(const LoopTree& loop_tree) { impl_->set(loop_tree); }

  template <typename T>
  inline T* data() const {
    return impl()->template data<T>();
  };

  inline void unify() const { const_cast<TensorImpl*>(impl_.get())->unify(); }
  inline void compile() const {
    const_cast<TensorImpl*>(impl_.get())->compile();
  }
  inline std::shared_ptr<Compiled> compiled() const {
    return const_cast<TensorImpl*>(impl_.get())->compiled();
  }

  inline bool has_deps() const { return impl_->deps_.size() > 0; }
};

}  // namespace lazy
}  // namespace loop_tool
