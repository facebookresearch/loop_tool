/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace loop_tool {
namespace symbolic {

inline uint64_t hash(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}

enum struct Op {
  // no inputs
  constant = 0,
  // unary
  negate,
  size,
  // binary
  add,
  multiply
};

inline int numInputs(const Op& op) {
  if (op < Op::negate) {
    return 0;
  } else if (op < Op::add) {
    return 1;
  }
  return 2;
}

struct Expr;

struct Symbol {
  // TODO replace with smaller construct
  std::string name_;
  int id_ = -1;
  Symbol() : id_(getNewId()), name_("X") {}
  Symbol(std::string name) : id_(getNewId()), name_(name) {}
  Symbol(const Symbol& s) : id_(s.id_), name_(s.name_) {}
  static const int getNewId();
  const int id() const;
  size_t hash() const;
  bool operator==(const Symbol& s) const;
  bool operator!=(const Symbol& s) const;
  const std::string& name() const;

  operator Expr() const;
  Expr operator+(const Symbol& rhs) const;
  Expr operator*(const Symbol& rhs) const;
  Expr operator+(const Expr& rhs) const;
  Expr operator*(const Expr& rhs) const;
};

struct Expr {
  enum class Type { value, symbol, function } type_;
  Op op_ = Op::constant;  // val_ and symbol_ are constant functions
  size_t val_;
  Symbol symbol_;
  std::vector<Expr> exprs_;
  explicit Expr(size_t val) : type_(Type::value), val_(val){};
  explicit Expr(int val) : Expr(static_cast<size_t>(val)){};
  explicit Expr(const Symbol& symbol) : type_(Type::symbol), symbol_(symbol){};
  explicit Expr(Op op, std::vector<Expr> exprs)
      : type_(Type::function), op_(op), exprs_(exprs){};
  Expr() = delete;
  size_t hash() const;
  size_t value() const;
  Symbol symbol() const;
  Op op() const;
  const std::vector<Expr>& args() const;
  Type type() const;
  // returns a new expr
  Expr walk(std::function<Expr(const Expr&)> f) const;
  Expr replace(Symbol A, Symbol B) const;
  Expr replace(Symbol A, size_t c) const;
  bool contains(Symbol s) const;

  static Expr size(const Expr& expr);
  Expr operator+(const Expr& rhs) const;
  Expr operator*(const Expr& rhs) const;
  bool operator!=(const Expr& rhs) const;
  bool operator==(const Expr& rhs) const;
  std::string dump() const;
};

// This might seem generic, but it should be limited to either:
//  - Expr(Symbol) -> Expr
//  - Expr::size(Symbol) -> Expr
using Constraint = std::pair<Expr, Expr>;

std::vector<Constraint> unify(std::vector<Constraint> constraints);

Expr differentiate(Expr, Symbol);

}  // namespace symbolic
}  // namespace loop_tool
