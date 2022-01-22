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
  divide
};

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
  int64_t val_;
  Symbol symbol_;
  std::vector<Expr> exprs_;
  mutable size_t hash_ = 0;
  mutable size_t symbol_hash_ = 0;
  void init();
  explicit Expr(int64_t val) : type_(Type::value), val_(val) { init(); };
  explicit Expr(const Symbol& symbol) : type_(Type::symbol), symbol_(symbol) {
    init();
  };
  explicit Expr(Op op, std::vector<Expr> exprs)
      : type_(Type::function), op_(op), exprs_(exprs) {
    init();
  };
  Expr() = delete;
  size_t hash(bool symbol_sensitive = false) const;
  int64_t value() const;
  Symbol symbol() const;
  Op op() const;
  const std::vector<Expr>& args() const;
  Type type() const;
  // returns a new expr
  Expr walk(std::function<Expr(const Expr&)> f) const;
  Expr replace(Symbol A, Symbol B) const;
  Expr replace(Symbol A, Expr e) const;
  Expr replace(const Expr& e, Symbol B) const;
  Expr replace(Symbol A, int64_t c) const;
  // actually returns count
  size_t contains(Symbol s) const;
  std::vector<Symbol> symbols(bool include_sized = true) const;

  static Expr size(const Expr& expr);
  static Expr max(const Expr& lhs, const Expr& rhs);
  Expr simplify() const;
  bool associative() const;
  bool can_evaluate() const;
  float evaluate() const;
  Expr operator+(const Expr& rhs) const;
  Expr operator*(const Expr& rhs) const;
  Expr operator-(const Expr& rhs) const;
  Expr operator-() const;
  Expr operator/(const Expr& rhs) const;
  Expr reciprocal() const;
  bool operator!=(const Expr& rhs) const;
  bool operator==(const Expr& rhs) const;
  std::string dump(bool short_form = false) const;
  size_t size() const;
};

// This might seem generic, but it should be limited to either:
//  - Expr(Symbol) -> Expr
//  - Expr::size(Symbol) -> Expr
using Constraint = std::pair<Expr, Expr>;

std::vector<Constraint> unify(std::vector<Constraint> constraints);
Constraint isolate(const Constraint& c, const Symbol& sym);

Expr differentiate(Expr, Symbol);

}  // namespace symbolic
}  // namespace loop_tool
