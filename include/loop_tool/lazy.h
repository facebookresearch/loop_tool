#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "loop_tool/backend.h"
#include "loop_tool/ir.h"
#include "loop_tool/symbolic.h"

namespace detail {

inline uint64_t hash(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}

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

template <typename... Args>
std::vector<Expr> Index(const Args&... args) {
  return std::vector<Expr>{Expr(args)...};
}

struct CachedCompilation {
  std::shared_ptr<Compiled> compilation;
  IR ir;
  LoopTree loop_tree;
  size_t output_size;
};

std::unordered_map<size_t, CachedCompilation>& getCompilationCache();

// wrapped by shared ptr for convenience
struct TensorImpl {
  size_t hash_ = 0;
  Operation op_ = Operation::constant;
  mutable void* data_ = nullptr;
  mutable bool owning_ = false;
  std::vector<Symbol> shape_;
  std::unordered_map<int, Expr> constraints_;
  std::vector<std::shared_ptr<TensorImpl>> deps_;

  void updateHash() {
    auto h = detail::hash((size_t)op_);
    h = detail::hash(h ^ shape_.size());
    for (const auto& s : shape_) {
      if (constraints_.count(s.id())) {
        // TODO Expr::hash()
        h = detail::hash(constraints_.at(s.id()).value());
      }
    }
    for (const auto& d : deps_) {
      h = detail::hash(h ^ d->hash());
    }
    hash_ = h;
  }

  size_t hash() const { return hash_; }

  ~TensorImpl() {
    if (owning_) {
      free(data_);
    }
  }

  TensorImpl(void* data, std::vector<size_t> sizes) : op_(Operation::constant) {
    bind(data, sizes);
    updateHash();
  }
  void bind(void* data, std::vector<size_t> sizes);

  TensorImpl(std::vector<Symbol> shape)
      : op_(Operation::constant), shape_(shape) {
    updateHash();
  }
  TensorImpl(Operation op, std::vector<Symbol> shape,
             std::vector<std::shared_ptr<TensorImpl>> deps)
      : op_(op), shape_(shape), deps_(deps) {
    updateHash();
  }

  inline const std::vector<Symbol>& shape() const { return shape_; }
  size_t size(int dim) const;

  // for these methods, int is Symbol::id
  void collectSymbolMap(std::unordered_map<int, Symbol>& symbol_map);
  void propagateSymbolMap(const std::unordered_map<int, Symbol>& symbol_map);
  void unifySymbols();
  void collectConstraints(std::vector<std::pair<int, Expr>>& constraints);
  void propogateConstraints(
      const std::unordered_map<int, Expr>& constraint_map);
  void unifyConstraints();
  void unify();
  void populateCompilationCache();

  std::vector<void*> getBuffers() const;

  template <typename T>
  T* data() const {
    if (owning_ && !data_) {
      size_t size = 1;
      for (auto i = 0; i < shape().size(); ++i) {
        ASSERT(constraints_.count(shape()[i].id()))
            << "cannot allocate owned tensor, size for " << shape()[i].name()
            << "not provided";
        size *= constraints_.at(shape()[i].id()).value();
      }
      data_ = malloc(sizeof(float) * size);
    }
    if (data_) {
      return static_cast<T*>(data_);
    }
    if (!getCompilationCache().count(hash())) {
      const_cast<TensorImpl*>(this)->populateCompilationCache();
    }
    auto& cc = getCompilationCache().at(hash());
    data_ = malloc(sizeof(float) * cc.output_size);
    owning_ = true;
    auto buffers = getBuffers();
    buffers.emplace_back(data_);
    cc.compilation->run(buffers, true);
    return static_cast<T*>(data_);
  };

  LoopTree loop_tree() const {
    if (deps_.size() == 0) {
      return LoopTree(IR());
    }
    (void)data<void>();
    auto& cc = getCompilationCache().at(hash());
    return cc.loop_tree;
  }

  LoopTree schedule(
      IR ir,
      const std::unordered_map<int, std::pair<IR::VarRef, size_t>>& var_map)
      const;
  IR::NodeRef resolve(
      IR& ir,
      std::unordered_map<int, std::pair<IR::VarRef, size_t>>& var_map) const;
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
            nullptr, std::vector<size_t>{static_cast<size_t>(sizes)...})) {}

  template <typename... Sizes,
            std::enable_if_t<detail::areT<size_t, Sizes...>::value, int> = 0>
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

  Tensor operator+(const Tensor& rhs) const {
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
        std::make_shared<TensorImpl>(Operation::add, new_shape, deps);
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

  template <typename... Args>
  Tensor to(const Args&... args) {
    std::vector<std::vector<Expr>> mapping{args...};
    return Tensor(impl_);
  }

  std::shared_ptr<TensorImpl> impl() const { return impl_; }
  std::vector<Symbol> shape() const { return impl_->shape(); }
  size_t size(int dim) const { return impl_->size(dim); }
  LoopTree loop_tree() const { return impl_->loop_tree(); }

  template <typename T>
  T* data() const {
    return impl()->template data<T>();
  };
};

}  // namespace lazy
}  // namespace loop_tool
