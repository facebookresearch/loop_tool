/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "error.h"
#include "smallvec.h"

namespace loop_tool {
namespace symbolic {

inline uint64_t hash(uint64_t x) {
  x += 1337;
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}

inline uint64_t hash_combine(uint64_t a, uint64_t b) {
  std::hash<uint64_t> hasher;
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t x = (hasher(b) ^ a) * kMul;
  x ^= (x >> 47);
  uint64_t y = (a ^ x) * kMul;
  y ^= (y >> 47);
  return y * kMul;
}

template <typename T>
struct Hash {
  std::size_t operator()(const T& k) const { return k.hash(); }
};

enum struct Op {
  // no inputs
  constant = 0,
  // unary
  negate,
  reciprocal,
  size,
  max,
  // binary
  add,
  multiply,
  divide,
  modulo
};

struct Expr;

struct Symbol {
  // TODO replace with smaller construct
  std::string name_;
  int32_t id_ = -1;
  Symbol() : id_(getNewId()), name_("X") {}
  Symbol(std::string name) : id_(getNewId()), name_(name) {}
  Symbol(const Symbol& s) : id_(s.id_), name_(s.name_) {}
  static const int32_t getNewId();
  const int32_t id() const;
  size_t hash() const;
  bool operator==(const Symbol& s) const;
  bool operator!=(const Symbol& s) const;
  std::string name() const;

  operator Expr() const;
  Expr operator+(const Symbol& rhs) const;
  Expr operator*(const Symbol& rhs) const;
  Expr operator+(const Expr& rhs) const;
  Expr operator*(const Expr& rhs) const;
};

struct Expr;

struct ExprImpl {
  enum class Type { value, symbol, function } type_;
  Op op_ = Op::constant;
  int64_t val_;
  Symbol symbol_;
  smallvec<std::shared_ptr<ExprImpl>, 2> args_;
  uint64_t hash_ = 0;
  uint64_t symbol_hash_ = 0;
  bool simplified_ = false;
  explicit ExprImpl(int64_t val)
      : type_(Type::value), val_(val), simplified_(true) {
    init();
  }
  explicit ExprImpl(const Symbol& symbol)
      : type_(Type::symbol), symbol_(symbol), simplified_(true) {
    init();
  }
  explicit ExprImpl(Op op, const Expr&, bool simplified = false);
  explicit ExprImpl(Op op, const Expr&, const Expr&, bool simplified = false);
  void init();
  inline uint64_t hash(bool symbol_sensitive) {
    if (symbol_sensitive) {
      return symbol_hash_;
    }
    return hash_;
  }

  bool contains(const Symbol& s) const {
    switch (type_) {
      case Type::symbol:
        if (symbol_ == s) {
          return true;
        }
        return false;
      case Type::function: {
        for (const auto& arg : args_) {
          if (arg->contains(s)) {
            return true;
          }
        }
      }
      default:
        return false;
    }
  }
};

struct Expr {
  std::shared_ptr<ExprImpl> impl_;
  using Type = ExprImpl::Type;

  explicit Expr() : impl_(std::make_shared<ExprImpl>(-1)) {}
  explicit Expr(std::shared_ptr<ExprImpl> impl) : impl_(impl) {}

  template <typename... Args>
  explicit Expr(Args... args)
      : impl_(std::make_shared<ExprImpl>(std::forward<Args>(args)...)) {}

  inline smallvec<Expr, 2> args() const {
    smallvec<Expr, 2> out;
    for (const auto& impl : impl_->args_) {
      out.emplace_back(Expr(impl));
    }
    return out;
  }

  Expr arg(int idx) const { return Expr(impl_->args_.at(idx)); }

  inline const smallvec<std::shared_ptr<ExprImpl>, 2>& impl_args() const {
    return impl_->args_;
  }

  bool simplified() const { return impl_->simplified_; }

  inline Type type() const { return impl_->type_; }
  inline Op op() const { return impl_->op_; }
  inline int64_t value() const {
    if (type() != Type::value) {
      ASSERT(type() == Type::value)
          << "attempted to get real value from symbolic or unsimplified "
             "expression: "
          << dump();
    }
    return impl_->val_;
  }

  inline const Symbol& symbol() const {
    if (type() != Type::symbol) {
      ASSERT(type() == Type::symbol)
          << "attempted to get symbol from value or unsimplified "
             "expression: "
          << dump();
    }
    return impl_->symbol_;
  }

  Expr walk(std::function<Expr(const Expr&)> f) const;
  void visit(std::function<void(const Expr&)> f) const;
  Expr replace(Symbol A, Symbol B) const;
  Expr replace(Symbol A, Expr e) const;
  Expr replace(const Expr& e, Symbol B) const;
  Expr replace(const Expr& e, int64_t c) const;
  Expr replace(Symbol A, int64_t c) const;

  std::string dump(bool short_form = false,
                   const std::unordered_map<Symbol, std::string, Hash<Symbol>>&
                       replacements = {}) const;

  inline uint64_t hash(bool symbol_sensitive = false) const {
    return impl_->hash(symbol_sensitive);
  }

  inline size_t contains(const Symbol& s) const { return impl_->contains(s); }
  std::vector<Symbol> symbols(bool include_sized = true) const;

  inline Expr operator+(const Expr& rhs) const {
    return Expr(Op::add, *this, rhs);
  }
  inline Expr operator*(const Expr& rhs) const {
    return Expr(Op::multiply, *this, rhs);
  }
  inline Expr operator-() const { return Expr(Op::negate, *this); }
  inline Expr operator-(const Expr& rhs) const { return *this + (-rhs); }
  inline Expr operator/(const Expr& rhs) const {
    return Expr(Op::divide, *this, rhs);
  }
  inline Expr operator%(const Expr& rhs) const {
    return Expr(Op::modulo, *this, rhs);
  }
  static Expr size(const Expr& arg) {
    ASSERT(arg.type() == Type::symbol);
    return Expr(Op::size, arg);
  }
  static Expr max(const Expr& lhs, const Expr& rhs) {
    return Expr(Op::max, lhs, rhs);
  }
  inline Expr reciprocal() const {
    if (type() == Type::value) {
      ASSERT(value() != 0) << "cannot calculate 1/0";
    }
    return Expr(Op::reciprocal, *this);
  }
  bool operator!=(const Expr& rhs) const;
  bool operator==(const Expr& rhs) const;
  Expr simplify() const;
  bool can_evaluate() const;
  float evaluate() const;
};

// This might seem generic, but it should be limited to either:
//  - Expr(Symbol) -> Expr
//  - Expr::size(Symbol) -> Expr
using Constraint = std::pair<Expr, Expr>;

std::vector<Constraint> unify(std::vector<Constraint> constraints);
bool can_isolate(const Constraint& c, const Symbol& sym);
Constraint isolate(const Constraint& c, const Symbol& sym);

std::vector<Constraint> evaluate(
    const std::vector<Constraint>& old_constraints);

Expr differentiate(Expr, Symbol);
// zero out every symbol
Expr intercept(Expr);

}  // namespace symbolic
}  // namespace loop_tool
