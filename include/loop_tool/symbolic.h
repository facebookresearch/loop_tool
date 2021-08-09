/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace loop_tool {
namespace symbolic {

enum struct Op {
  // no inputs
  constant,
  // unary
  negate,
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

const int getNewSymbolId();

struct Symbol {
  // TODO replace with smaller construct
  std::string name_;
  int id_ = -1;
  Symbol() : id_(getNewSymbolId()), name_("X") {}
  Symbol(std::string name) : id_(getNewSymbolId()), name_(name) {}
  Symbol(const Symbol& s) : id_(s.id_), name_(s.name_) {}
  const int id() const { return id_; }
  bool operator==(const Symbol& s) const { return s.id() == id_; }
  const std::string& name() const { return name_; }
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
  size_t value() const;
  Symbol symbol() const;
  Op op() const;
  const std::vector<Expr>& args() const;
  Type type() const;
  Expr operator+(const Expr& rhs) const;
  Expr operator*(const Expr& rhs) const;
  bool operator!=(const Expr& rhs) const;
  bool operator==(const Expr& rhs) const;
  std::string dump() const;
};

// Symbol::id -> Expr
std::vector<std::pair<int, Expr>> unify(
    std::vector<std::pair<int, Expr>> constraints);

}  // namespace symbolic
}  // namespace loop_tool
