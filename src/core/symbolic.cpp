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
size_t Symbol::hash() const { return symbolic::hash(id_); }
bool Symbol::operator==(const Symbol& s) const { return s.id() == id_; }
bool Symbol::operator!=(const Symbol& s) const { return !(s.id() == id_); }
const std::string& Symbol::name() const { return name_; }

Symbol::operator Expr() const { return Expr(*this); }
Expr Symbol::operator+(const Symbol& rhs) const {
  return Expr(*this) + Expr(rhs);
}
Expr Symbol::operator*(const Symbol& rhs) const {
  return Expr(*this) * Expr(rhs);
}

Expr Symbol::operator+(const Expr& rhs) const { return Expr(*this) + rhs; }
Expr Symbol::operator*(const Expr& rhs) const { return Expr(*this) * rhs; }

size_t Expr::hash() const {
  size_t h = symbolic::hash((int)op_);
  if (type_ == Type::value) {
    h = symbolic::hash(h ^ symbolic::hash(val_));
  } else if (type_ == Type::symbol) {
    h = symbolic::hash(h ^ symbol().hash());
  }
  for (const auto& expr : exprs_) {
    h = symbolic::hash(h ^ expr.hash());
  }
  return h;
}

size_t Expr::value() const {
  ASSERT(type_ == Expr::Type::value)
      << "attempted to get real value from symbolic or unsimplified expression";
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

Expr Expr::replace(Symbol A, Symbol B) const {
  switch (type()) {
    case Expr::Type::symbol:
      if (symbol() == A) {
        return B;
      }
      return symbol();
    case Expr::Type::value:
      return *this;
    case Expr::Type::function: {
      std::vector<Expr> new_args;
      for (const auto& arg : args()) {
        new_args.emplace_back(arg.replace(A, B));
      }
      return Expr(op(), new_args);
    }
    default:
      ASSERT(0) << "couldn't process replacement!";
      return B;
  }
}

Expr Expr::replace(Symbol A, size_t c) const {
  switch (type()) {
    case Expr::Type::symbol:
      if (symbol() == A) {
        return Expr(c);
      }
      return symbol();
    case Expr::Type::value:
      return *this;
    case Expr::Type::function: {
      std::vector<Expr> new_args;
      for (const auto& arg : args()) {
        new_args.emplace_back(arg.replace(A, c));
      }
      return Expr(op(), new_args);
    }
    default:
      ASSERT(0) << "couldn't process replacement!";
      return Expr(c);
  }
}

Expr Expr::walk(std::function<Expr(const Expr&)> f) const {
  if (type() == Expr::Type::function) {
    std::vector<Expr> new_args;
    for (const auto& arg : args()) {
      new_args.emplace_back(arg.walk(f));
    }
    return f(Expr(op(), new_args));
  }
  return f(*this);
}

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
    ss << symbol().name() << "[id:" << symbol().id() << "]";
  } else if (op_ == Op::size) {
    ASSERT(args().size() == 1);
    ss << "|" << args().at(0).dump() << "|";
  } else {
    ASSERT(args().size() == 2);
    auto lhs = args().at(0);
    auto rhs = args().at(1);
    if (lhs.op() == Op::constant || lhs.args().size() == 1) {
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

    if (rhs.op() == Op::constant || rhs.args().size() == 1) {
      ss << rhs.dump();
    } else {
      ss << "(" << rhs.dump() << ")";
    }
  }
  return ss.str();
}

// TODO: AC unification algorithm i.e. Expr = Expr constraints with
// associativity/commutativity
std::vector<Constraint> unify(std::vector<Constraint> constraints) {
  std::unordered_map<int, Symbol> all_symbols;

  auto is_simple_size_expr = [&](const Expr& expr) {
    return (expr.op() == Op::size) && (expr.args().size() == 1) &&
           (expr.args().at(0).type() == Expr::Type::symbol);
  };

  // purely a sanity check
  for (const auto& constraint : constraints) {
    const auto& expr = constraint.first;
    ASSERT(expr.type() == Expr::Type::symbol || is_simple_size_expr(expr))
        << "cannot unify constraint " << expr.dump() << " = "
        << constraint.second.dump();
    auto symbol = [&]() {
      if (is_simple_size_expr(expr)) {
        return expr.args().at(0).symbol();
      }
      return expr.symbol();
    }();
    all_symbols.emplace(symbol.id(), symbol);
  }

  std::function<Expr(Expr)> eval_expr;
  // Symbol.id() -> value
  std::unordered_map<int, Expr> replacements;
  std::unordered_map<int, Expr> size_replacements;

  auto pass = [&]() -> bool {
    bool updated = false;
    for (auto& p : constraints) {
      p.second = eval_expr(p.second);
      // we've proven an identity, no need to process it
      if (p.first == p.second) {
        continue;
      }
      // same logic for either updating Symbol or size(Symbol)
      auto update_replacements = [&](int id,
                                     std::unordered_map<int, Expr>& reps) {
        if (!reps.count(id)) {
          reps.emplace(id, p.second);
          updated = true;
        } else if (reps.at(id) != p.second) {
          auto old = reps.at(id);
          ASSERT(old != p.second);
          ASSERT(old.type() != Expr::Type::value)
              << "mismatched values for " << p.first.dump() << ": "
              << old.dump() << " vs " << p.second.dump();
          reps.at(id) = p.second;
          updated = true;
        }
      };

      if (p.first.type() == Expr::Type::symbol) {
        auto id = p.first.symbol().id();
        update_replacements(id, replacements);
      } else {
        ASSERT(is_simple_size_expr(p.first));
        auto id = p.first.args().at(0).symbol().id();
        update_replacements(id, size_replacements);
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
    } else if (is_simple_size_expr(e)) {
      auto id = e.args().at(0).symbol().id();
      if (size_replacements.count(id)) {
        return size_replacements.at(id);
      }
      return e;
    }
    ASSERT(e.type() == Expr::Type::function);
    if (e.args().size() == 2) {
      auto lhs = eval_expr(e.args().at(0));
      auto rhs = eval_expr(e.args().at(1));
      if (e.op() == Op::add) {
        return lhs + rhs;
      } else if (e.op() == Op::multiply) {
        return lhs * rhs;
      }
      ASSERT(0) << "unknown expression op";
    }
    ASSERT(0) << "unknown expression op";
    return e;
  };

  while (pass())
    ;

  // For unsized symbols (i.e. size(Symbol) = ?),
  // we can use the indexing equations to resolve them.
  // Solve for the last element in the iteration:
  //  |X| - 1 = expr() // all symbols replace with size(Symbol) - 1
  // and then we have the size
  //  |X| = expr() + 1
  //
  // e.g. convolving
  //  X = Y + K
  //  |K| = 3
  //  |Y| = 12
  //  |X| - 1 = 11 + 2
  //  |X| = 13
  // e.g. flattening
  //  X = A * |B| + B
  //  |B| = 8
  //  |A| = 8
  //  |X| - 1 = 7 * 8 + 7
  //  |X| = 64
  auto derive_size_function = [&](const Symbol& s) {
    auto expr = replacements.at(s.id());

    // Find symbol make up
    std::vector<Symbol> composed_symbols;
    expr.walk([&](const Expr& e) {
      if (e.type() == Expr::Type::symbol) {
        composed_symbols.emplace_back(e.symbol());
        ASSERT(s != e.symbol()) << "impossible constraint found";
      }
      return e;
    });

    for (const auto& cs : composed_symbols) {
      if (!size_replacements.count(cs.id())) {
        return false;
      }

      auto size_expr = size_replacements.at(cs.id());
      if (size_expr.type() != Expr::Type::value) {
        return false;
      }
      expr = expr.replace(cs, size_expr.value() - 1);
    }

    auto size_expr = eval_expr(expr + Expr(1));
    size_replacements.emplace(s.id(), size_expr);
    return true;
  };

  auto size_pass = [&]() {
    auto updated = false;
    for (const auto& p : constraints) {
      if (p.first.type() == Expr::Type::symbol) {
        auto symbol = p.first.symbol();
        if (!size_replacements.count(symbol.id())) {
          updated |= derive_size_function(symbol);
        }
      }
    }
    return updated;
  };

  while (size_pass())
    ;

  std::vector<Constraint> out;
  for (const auto& p : all_symbols) {
    auto symbol = p.second;
    out.emplace_back(std::make_pair(symbol, eval_expr(symbol)));
    out.emplace_back(
        std::make_pair(Expr::size(symbol), eval_expr(Expr::size(symbol))));
  }
  return out;
}

}  // namespace symbolic
}  // namespace loop_tool
