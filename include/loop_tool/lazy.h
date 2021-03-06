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

struct CachedLowered {
  IR ir;
  LoopTree loop_tree;
  int64_t output_size;
  std::vector<int64_t> sizes;
};

std::unordered_map<size_t, std::shared_ptr<Compiled>>& getCompilationCache();
std::unordered_map<size_t, CachedLowered>& getLoweredCache();

// wrapped by shared ptr for convenience
struct TensorImpl {
  uint64_t hash_ = 0;
  bool unified_ = false;
  Operation op_ = Operation::constant;
  mutable Memory memory_;
  mutable bool owning_ = true;
  mutable bool force_recompute_ = false;
  std::vector<Symbol> shape_;
  mutable std::vector<int64_t> sizes_;
  std::vector<Constraint> constraints_;
  std::vector<std::shared_ptr<TensorImpl>> deps_;

  std::unordered_map<int, Expr> size_constraints() const {
    std::unordered_map<int, Expr> out;
    for (const auto& c : constraints_) {
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

  void updateHash(bool force_unify = false) {
    if (force_unify) {
      const_cast<TensorImpl*>(this)->unify(force_unify);
    }
    auto h = symbolic::hash((size_t)op_);
    h = symbolic::hash_combine(h, symbolic::hash(shape_.size()));
    auto cm = constraints();
    // TODO known error with subtlely similar constraints but disimilar layouts
    for (auto& p : cm) {
      h = symbolic::hash_combine(h, p.first.hash());
      h = symbolic::hash_combine(h, p.second.hash());
    }
    std::unordered_map<symbolic::Symbol, int, symbolic::Hash<symbolic::Symbol>>
        unique_syms;
    auto hash_sym = [&](const symbolic::Symbol& sym) {
      if (op_ == Operation::name) {
        return symbolic::hash(0);
      }
      if (!unique_syms.count(sym)) {
        unique_syms[sym] = unique_syms.size();
      }
      return symbolic::hash(unique_syms.at(sym));
    };
    for (const auto& d : deps_) {
      h = symbolic::hash_combine(h, d->hash());
      for (const auto& sym : d->shape()) {
        h = symbolic::hash_combine(h, hash_sym(sym));
      }
    }
    for (const auto& sym : shape()) {
      h = symbolic::hash_combine(h, hash_sym(sym));
    }
    hash_ = h;
  }

  inline uint64_t hash() const { return hash_; }
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
    for (const auto& p : constraints) {
      p.first.walk([&](const Expr& e) {
        if (e.type() == Expr::Type::symbol) {
          sym_lhs[e.symbol().id()] = e.symbol();
        }
        return e;
      });
    }
    // assume all input shapes are calculated
    for (const auto& dep : deps_) {
      for (const auto& sym : dep->shape()) {
        sym_lhs[sym.id()] = sym;
      }
    }
    for (const auto& sym : shape_) {
      if (sym_lhs.count(sym.id())) {
        continue;
      }
      for (auto& c : constraints) {
        c.second.walk([&](const Expr& e) {
          if (e.type() == Expr::Type::symbol &&
              sym_lhs.count(e.symbol().id()) == 0 &&
              can_isolate(c, e.symbol())) {
            auto isolated_constraint = isolate(c, e.symbol());
            constraints_.emplace_back(isolated_constraint);
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
  void collectSymbolMap(std::unordered_map<int, Symbol>& symbol_map,
                        std::unordered_set<TensorImpl*>& seen);
  void propagateSymbolMap(const std::unordered_map<int, Symbol>& symbol_map,
                          std::unordered_set<TensorImpl*>& seen);
  void unifySymbols();
  void collectConstraints(std::vector<Constraint>& constraints,
                          std::unordered_set<TensorImpl*>& seen,
                          std::unordered_set<int64_t>& seen_constraint_hashes);
  void propagateConstraints(const std::vector<Constraint>& constraints,
                            std::unordered_set<TensorImpl*>& seen);
  void unifyConstraints();
  void unify(bool force = false);
  void populateCompilationCache();
  void populateLoweredCache();

  std::vector<void*> getInputBuffers(
      std::unordered_set<const TensorImpl*>& seen) const;

  inline void force_recompute() const { force_recompute_ = true; }
  inline void clear_cache() const {
    auto h = hash();
    auto& cache = getCompilationCache();
    if (cache.count(h)) {
      cache.erase(h);
    }
    ASSERT(getCompilationCache().count(h) == 0);
    auto& lcache = getLoweredCache();
    if (lcache.count(h)) {
      lcache.erase(h);
    }
    ASSERT(getLoweredCache().count(h) == 0);
  }

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
    auto& lowered = getLoweredCache().at(hash());

    if (owning_) {
      memory_ = alloc(sizeof(float) * lowered.output_size);
    } else {
      ASSERT(memory_.address);
    }

    std::unordered_set<const TensorImpl*> seen;
    auto buffers = getInputBuffers(seen);
    buffers.emplace_back(memory_.address);
    cc->run(buffers, true);
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
    if (getLoweredCache().count(h)) {
      auto& lowered = getLoweredCache().at(h);
      return lowered.ir;
    }
    auto ll = lower();
    return schedule(ll.first, ll.second).ir;
  }

  inline LoopTree loop_tree() const {
    auto h = hash();
    if (getLoweredCache().count(h)) {
      auto& lowered = getLoweredCache().at(h);
      return lowered.loop_tree;
    }
    auto ll = lower();
    return schedule(ll.first, ll.second);
  }

  inline void compile() {
    if (!getCompilationCache().count(hash())) {
      populateCompilationCache();
    }
  }

  inline std::string code() const { return compiled()->dump(); }

  inline void set(const IR& ir) {
    auto h = hash();
    clear_cache();
    if (!getLoweredCache().count(h)) {
      const_cast<TensorImpl*>(this)->populateLoweredCache();
    }
    auto& lowered = getLoweredCache().at(h);
    LoopTree loop_tree(ir);
    lowered = CachedLowered{ir, loop_tree, lowered.output_size, lowered.sizes};
  }

  inline void set(const LoopTree& loop_tree) {
    auto h = hash();
    clear_cache();
    if (!getLoweredCache().count(h)) {
      const_cast<TensorImpl*>(this)->populateLoweredCache();
    }
    auto& lowered = getLoweredCache().at(h);
    lowered = CachedLowered{loop_tree.ir, loop_tree, lowered.output_size,
                            lowered.sizes};
  }

  inline std::shared_ptr<Compiled> compiled() const {
    auto h = hash();
    if (!getCompilationCache().count(h)) {
      const_cast<TensorImpl*>(this)->compile();
    }
    auto& cc = getCompilationCache().at(h);
    return cc;
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
  template <typename... Sizes,
            std::enable_if_t<detail::areT<size_t, Sizes...>::value, int> = 0>
  Tensor(Sizes... sizes)
      : impl_(std::make_shared<TensorImpl>(
            nullptr, std::vector<int64_t>{static_cast<int64_t>(sizes)...})) {}

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

  Tensor min(const Tensor& rhs) const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_, rhs.impl()};
    auto new_impl = std::make_shared<TensorImpl>(Operation::min,
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

  Tensor pad(Symbol padded_dim, int64_t pre, int64_t post) const {
    ASSERT(pre >= 0) << "cannot pad by a negative number";
    ASSERT(post >= 0) << "cannot pad by a negative number";
    if (pre == 0 && post == 0) {
      ASSERT(0) << "unecessary pad operation (padded by zero)";
      return *this;
    }
    auto new_sym = Symbol(padded_dim.name() + "_p_" + std::to_string(pre) +
                          "_" + std::to_string(post));
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

  Tensor pad(Symbol padded_dim, int64_t amount) const {
    return pad(padded_dim, amount, amount);
  }

  Tensor transpose(std::vector<int> order) const {
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

  Tensor transpose(std::vector<Symbol> new_shape) const {
    ASSERT(new_shape.size() == shape().size()) << "invalid transpose";
    auto new_impl = std::make_shared<TensorImpl>(*impl_);
    new_impl->shape_ = new_shape;
    return Tensor(new_impl);
  }

  Tensor flatten(std::vector<Symbol> symbols, Symbol new_symbol) const {
    ASSERT(symbols.size() >= 1);
    auto contained = to_set<Symbol, symbolic::Hash>(symbols);
    std::vector<Symbol> new_shape;
    std::vector<Symbol> output_shape;
    bool contiguous = false;
    bool ended = false;
    for (const auto& sym : shape()) {
      if (contained.count(sym)) {
        ASSERT(!ended) << "flatten only works on contiguous dimensions. "
                          "Otherwise, transpose first.";
        contiguous = true;
        continue;
      }
      if (contiguous) {
        ended = true;
      }
      new_shape.emplace_back(sym);
      output_shape.emplace_back(sym);
    }
    output_shape.emplace_back(new_symbol);
    for (const auto& sym : symbols) {
      new_shape.emplace_back(sym);
    }
    std::reverse(symbols.begin(), symbols.end());
    auto flattened_dim = Expr(0);
    auto running_size = Expr(1);
    for (const auto& sym : symbols) {
      flattened_dim = sym * running_size + flattened_dim;
      running_size = Expr::size(sym) * running_size;
    }
    return to(output_shape, Constraint(new_symbol, flattened_dim));
  }

  Tensor reduce(Operation op, const std::vector<Symbol>& reduction_vars) const {
    ASSERT(reduction_vars.size()) << "reduction variables required (got none)";
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
    ASSERT(new_shape.size() != shape().size())
        << "reduction variables not over any input";
    return Tensor(std::make_shared<TensorImpl>(op, new_shape, deps));
  }

  Tensor sum(const std::vector<Symbol>& reduction_vars) const {
    return reduce(Operation::add, reduction_vars);
  }

  Tensor prod(const std::vector<Symbol>& reduction_vars) const {
    return reduce(Operation::multiply, reduction_vars);
  }

  Tensor max(const std::vector<Symbol>& reduction_vars) const {
    return reduce(Operation::max, reduction_vars);
  }

  Tensor min(const std::vector<Symbol>& reduction_vars) const {
    return reduce(Operation::min, reduction_vars);
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

  Tensor log() const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    auto new_impl =
        std::make_shared<TensorImpl>(Operation::log, impl_->shape(), deps);
    return Tensor(new_impl);
  }

  Tensor sqrt() const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    auto new_impl =
        std::make_shared<TensorImpl>(Operation::sqrt, impl_->shape(), deps);
    return Tensor(new_impl);
  }

  Tensor abs() const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    auto new_impl =
        std::make_shared<TensorImpl>(Operation::abs, impl_->shape(), deps);
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

  inline Tensor as(std::vector<Symbol> shape) const {
    std::vector<std::shared_ptr<TensorImpl>> deps{impl_};
    return Tensor(std::make_shared<TensorImpl>(Operation::name, shape, deps));
  }

  template <typename... Args>
  Tensor as(const Args&... args) const {
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
    for (auto s : sizes()) {
      total *= s;
    }
    return total;
  }
  inline LoopTree loop_tree() const { return impl_->loop_tree(); }
  inline IR ir() const { return impl_->ir(); }
  inline std::string code() const { return impl_->code(); }
  inline void set(const IR& ir) { impl_->set(ir); }
  inline void set(const LoopTree& loop_tree) { impl_->set(loop_tree); }

  inline void force_recompute() const { impl()->force_recompute(); }
  inline void clear_cache() const { impl()->clear_cache(); }
  inline size_t hash() const { return impl()->hash(); }

  template <typename T>
  inline T* data() const {
    return impl()->template data<T>();
  };

  inline void unify(bool force = false) const {
    const_cast<TensorImpl*>(impl_.get())->unify(force);
  }
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
