/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/symbolic.h"

#include <functional>
#include <unordered_map>

#include "loop_tool/error.h"

namespace loop_tool {
namespace symbolic {

const int Symbol::getNewId() {
  static int symbol_count_ = 0;
  return symbol_count_++;
}

const int Symbol::id() const { return id_; }
bool Symbol::operator==(const Symbol& s) const { return s.id() == id_; }
const std::string& Symbol::name() const { return name_; }

Symbol::operator Expr() const { return Expr(*this); }
Expr Symbol::operator+(const Symbol& rhs) const {
  return Expr(*this) + Expr(rhs);
}
Expr Symbol::operator*(const Symbol& rhs) const {
  return Expr(*this) * Expr(rhs);
}

size_t Expr::value() const {
  ASSERT(type_ == Expr::Type::value);
  return val_;
}

Symbol Expr::symbol() const {
  ASSERT(type_ == Expr::Type::symbol);
  return symbol_;
}

Op Expr::op() const { return op_; }

const std::vector<Expr>& Expr::args() const {
  ASSERT(type_ == Expr::Type::function);
  return exprs_;
}

Expr::Type Expr::type() const { return type_; }

Expr Expr::size(const Expr& expr) {
  ASSERT(expr.type() == Expr::Type::symbol) << "size() only works on symbols";
  return Expr(Op::size, {expr});
}

Expr Expr::operator+(const Expr& rhs) const {
  if (type() == Expr::Type::value) {
    if (rhs.type() == Expr::Type::value) {
      return Expr(value() + rhs.value());
    }
  }
  return Expr(Op::add, {*this, rhs});
}

Expr Expr::operator*(const Expr& rhs) const {
  if (type() == Expr::Type::value) {
    if (rhs.type() == Expr::Type::value) {
      return Expr(value() * rhs.value());
    }
  }
  return Expr(Op::multiply, {*this, rhs});
}

bool Expr::operator!=(const Expr& rhs) const { return !(*this == rhs); }

bool Expr::operator==(const Expr& rhs) const {
  if (type_ == Expr::Type::value) {
    return rhs.type() == Expr::Type::value && rhs.value() == value();
  } else if (type_ == Expr::Type::symbol) {
    return rhs.type() == Expr::Type::symbol && rhs.symbol() == symbol();
  }
  ASSERT(type_ == Expr::Type::function);
  if (rhs.type() != Expr::Type::function) {
    return false;
  }
  bool match = true;
  if (args().size() == rhs.args().size()) {
    for (auto i = 0; i < args().size(); ++i) {
      match &= args().at(i) == rhs.args().at(i);
    }
  } else {
    match = false;
  }
  return rhs.op() == op() && match;
}

std::string Expr::dump() const {
  std::stringstream ss;
  if (type_ == Expr::Type::value) {
    ss << value();
  } else if (type_ == Expr::Type::symbol) {
    ss << symbol().name();
  } else {
    ASSERT(args().size() == 2);
    auto lhs = args().at(0);
    auto rhs = args().at(1);
    if (lhs.op() == Op::constant) {
      ss << lhs.dump();
    } else {
      ss << "(" << lhs.dump() << ")";
    }

    if (op_ == Op::add) {
      ss << "+";
    } else if (op_ == Op::multiply) {
      ss << "*";
    } else {
      ASSERT(0) << "can't print this op";
    }

    if (rhs.op() == Op::constant) {
      ss << rhs.dump();
    } else {
      ss << "(" << rhs.dump() << ")";
    }
  }
  return ss.str();
}

// TODO: AC unification algorithm i.e. Expr = Expr constraints with
// associativity/commutativity
std::vector<std::pair<int, Expr>> unify(
    std::vector<std::pair<int, Expr>> constraints) {
  std::function<Expr(Expr)> eval_expr;
  // Symbol.id() -> value
  std::unordered_map<int, Expr> replacements;
  auto pass = [&]() -> bool {
    bool updated = false;
    for (auto& p : constraints) {
      p.second = eval_expr(p.second);
      if (!replacements.count(p.first)) {
        replacements.emplace(p.first, p.second);
        updated = true;
      }
      if (replacements.at(p.first) != p.second) {
        replacements.at(p.first) = p.second;
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
    if (e.op() == Op::add) {
      return lhs + rhs;
    } else if (e.op() == Op::multiply) {
      return lhs * rhs;
    }
    ASSERT(0) << "unknown expression op";
    return e;
  };

  while (pass())
    ;

  std::vector<std::pair<int, Expr>> out;
  for (const auto& p : constraints) {
    out.emplace_back(std::make_pair(p.first, eval_expr(p.second)));
  }
  return out;
}

}  // namespace symbolic
}  // namespace loop_tool
