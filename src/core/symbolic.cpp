/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/symbolic.h"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

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

size_t Expr::hash(bool symbol_sensitive) const {
  size_t h = symbolic::hash((int)op_);
  if (type_ == Type::value) {
    h = symbolic::hash(h ^ symbolic::hash(val_));
  } else if (type_ == Type::symbol) {
    // for exprs, we usually pretend all symbols are the same
    if (symbol_sensitive) {
      h = symbolic::hash(h ^ symbol().hash());
    } else {
      h = symbolic::hash(h ^ symbolic::hash(1337));
    }
  }
  for (const auto& expr : exprs_) {
    h = symbolic::hash(h ^ expr.hash());
  }
  return h;
}

int64_t Expr::value() const {
  if (type_ != Expr::Type::value) {
    ASSERT(type_ == Expr::Type::value)
        << "attempted to get real value from symbolic or unsimplified "
           "expression: "
        << dump();
  }
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

Expr Expr::replace(Symbol A, Expr e) const {
  switch (type()) {
    case Expr::Type::symbol:
      if (symbol() == A) {
        return e;
      }
      return symbol();
    case Expr::Type::value:
      return *this;
    case Expr::Type::function: {
      std::vector<Expr> new_args;
      for (const auto& arg : args()) {
        new_args.emplace_back(arg.replace(A, e));
      }
      return Expr(op(), new_args);
    }
    default:
      ASSERT(0) << "couldn't process replacement!";
      return e;
  }
}
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

Expr Expr::replace(const Expr& e, Symbol B) const {
  if (*this == e) {
    return B;
  }
  switch (type()) {
    case Expr::Type::symbol:
    case Expr::Type::value:
      return *this;
    case Expr::Type::function: {
      std::vector<Expr> new_args;
      for (const auto& arg : args()) {
        new_args.emplace_back(arg.replace(e, B));
      }
      return Expr(op(), new_args);
    }
    default:
      ASSERT(0) << "couldn't process replacement!";
      return B;
  }
}

Expr Expr::replace(Symbol A, int64_t c) const {
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

size_t Expr::contains(Symbol s) const {
  switch (type()) {
    case Expr::Type::symbol:
      if (symbol() == s) {
        return 1;
      }
      return 0;
    case Expr::Type::function: {
      size_t count = 0;
      for (const auto& arg : args()) {
        count += arg.contains(s);
      }
      return count;
    }
    default:
      return 0;
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
  return Expr(Op::size, {expr}).simplify();
}

Expr Expr::max(const Expr& lhs, const Expr& rhs) {
  return Expr(Op::max, {lhs, rhs}).simplify();
}

Expr Expr::simplify() const {
  if (type() != Expr::Type::function) {
    return *this;
  }
  auto sorted_args = args();
  for (auto& arg : sorted_args) {
    arg = arg.simplify();
  }
  std::sort(
      sorted_args.begin(), sorted_args.end(),
      [](const Expr& a, const Expr& b) { return a.hash(true) > b.hash(true); });
  switch (op()) {
    case Op::add: {
      auto lhs = sorted_args.at(0);
      auto rhs = sorted_args.at(1);
      if (lhs.type() == Expr::Type::value) {
        if (rhs.type() == Expr::Type::value) {
          return Expr(lhs.value() + rhs.value());
        }
        if (lhs.value() == 0) {
          return rhs;
        }
      }
      if (rhs.type() == Expr::Type::value) {
        if (rhs.value() == 0) {
          return lhs;
        }
      }
      return Expr(op(), sorted_args);
    }
    case Op::multiply: {
      auto lhs = sorted_args.at(0);
      auto rhs = sorted_args.at(1);
      if (lhs.type() == Expr::Type::value) {
        if (rhs.type() == Expr::Type::value) {
          return Expr(lhs.value() * rhs.value());
        }
        if (lhs.value() == 1) {
          return rhs;
        }
        if (lhs.value() == 0) {
          return Expr(0);
        }
      }
      if (rhs.type() == Expr::Type::value) {
        if (rhs.value() == 0) {
          return Expr(0);
        }
        if (rhs.value() == 1) {
          return lhs;
        }
      }
      return Expr(op(), sorted_args);
    }
    case Op::divide: {
      auto lhs = sorted_args.at(0);
      auto rhs = sorted_args.at(1);
      if (lhs.type() == Expr::Type::value) {
        if (rhs.type() == Expr::Type::value) {
          if (lhs.value() % rhs.value() == 0) {
            return Expr(lhs.value() / rhs.value());
          }
        }
      }
      if (rhs.type() == Expr::Type::value && rhs.value() == 1) {
        return lhs;
      }
      return Expr(op(), sorted_args);
    }
    case Op::max: {
      auto lhs = sorted_args.at(0);
      auto rhs = sorted_args.at(1);
      if (lhs.type() == Expr::Type::value) {
        if (rhs.type() == Expr::Type::value) {
          return Expr(std::max(lhs.value(), rhs.value()));
        }
        if (lhs.value() == std::numeric_limits<decltype(lhs.value())>::min()) {
          return rhs;
        }
      }
      if (lhs == rhs) {
        return lhs;
      }
      return Expr(op(), sorted_args);
    }
    case Op::negate: {
      const auto& arg = sorted_args.at(0);
      if (arg.type() == Expr::Type::value) {
        return Expr(-arg.value());
      }
      if (arg.type() == Expr::Type::function && arg.op() == Op::negate) {
        return arg.args().at(0).simplify();
      }
      return Expr(op(), sorted_args);
    }
    default: {
      return Expr(op(), sorted_args);
    }
  };
  ASSERT(0);
  return *this;
}

Expr Expr::operator+(const Expr& rhs) const {
  return Expr(Op::add, {*this, rhs}).simplify();
}

Expr Expr::operator*(const Expr& rhs) const {
  return Expr(Op::multiply, {*this, rhs}).simplify();
}

Expr Expr::operator-() const { return Expr(Op::negate, {*this}).simplify(); }

Expr Expr::operator-(const Expr& rhs) const { return *this + (-rhs); }

Expr Expr::reciprocal() const {
  if (type() == Expr::Type::value) {
    ASSERT(value() != 0) << "cannot calculate 1/0";
  }
  return Expr(Op::reciprocal, {*this});
}

Expr Expr::operator/(const Expr& rhs) const {
  return Expr(Op::divide, {*this, rhs}).simplify();
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
    auto sorted_args = args();
    auto sorted_rhs_args = rhs.args();
    auto sort_fn = [](const Expr& a, const Expr& b) {
      return a.hash() < b.hash();
    };
    std::sort(sorted_args.begin(), sorted_args.end(), sort_fn);
    std::sort(sorted_rhs_args.begin(), sorted_rhs_args.end(), sort_fn);
    for (auto i = 0; i < sorted_args.size(); ++i) {
      match &= sorted_args.at(i) == sorted_rhs_args.at(i);
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
  } else if (op_ == Op::max) {
    ASSERT(args().size() == 2);
    ss << "max(" << args().at(0).dump() << ", " << args().at(1).dump() << ")";
  } else if (op_ == Op::negate) {
    ASSERT(args().size() == 1);
    ss << "-" << args().at(0).dump();
  } else if (op_ == Op::reciprocal) {
    ASSERT(args().size() == 1);
    ss << args().at(0).dump() << "^-1";
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
    } else if (op_ == Op::divide) {
      ss << "/";
    } else {
      ASSERT(0) << "can't print this op id " << (int)op_;
    }

    if (rhs.op() == Op::constant || rhs.args().size() == 1) {
      ss << rhs.dump();
    } else {
      ss << "(" << rhs.dump() << ")";
    }
  }
  return ss.str();
}

size_t Expr::size() const {
  size_t s = 0;
  walk([&](const Expr& e) {
    s++;
    return e;
  });
  return s;
}

bool can_isolate(const Expr& e, const Symbol& sym) {
  if (e.type() != Expr::Type::function) {
    return true;
  }
  if (!e.contains(sym)) {
    return true;
  }
  switch (e.op()) {
    case Op::add:
    case Op::multiply:
    case Op::negate:
    case Op::reciprocal:
    case Op::divide: {
      bool res = true;
      for (const auto& arg : e.args()) {
        res &= can_isolate(arg, sym);
      }
      return res;
    }
    default:
      break;
  }
  return false;
}

bool can_isolate(const Constraint& c, const Symbol& sym) {
  const auto& lhs = c.first;
  const auto& rhs = c.second;
  if (lhs.contains(sym) + rhs.contains(sym) > 1) {
    return false;
  }
  return can_isolate(lhs, sym) && can_isolate(rhs, sym);
}

// take a constraint and move everything besides the sym
// to the rhs
// TODO This is extremely primitive right now!
// Any contribution of testing/impl would be highly appreciated :)
Constraint isolate(const Constraint& c, const Symbol& sym) {
  const auto& lhs = c.first;
  const auto& rhs = c.second;
  if (!lhs.contains(sym) && rhs.contains(sym)) {
    return isolate(std::make_pair(rhs, lhs), sym);
  }
  ASSERT(lhs.contains(sym) && !rhs.contains(sym))
      << "cannot isolate with variable on both rhs and lhs of constraint yet: "
      << lhs.dump() << " = " << rhs.dump() << " for sym " << Expr(sym).dump();
  if (lhs == Expr(sym)) {
    return std::make_pair(lhs, rhs);
  }

  if (lhs.type() == Expr::Type::function) {
    ASSERT(can_isolate(lhs, sym))
        << "cannot isolate through " << lhs.dump()
        << ", you may need to update the can_isolate function";
    switch (lhs.op()) {
      case Op::add: {
        auto llhs = lhs.args().at(0);
        auto lrhs = lhs.args().at(1);
        if (llhs.contains(sym)) {
          return isolate(std::make_pair(llhs, rhs - lrhs), sym);
        }
        return isolate(std::make_pair(lrhs, rhs - llhs), sym);
      }
      case Op::multiply: {
        auto llhs = lhs.args().at(0);
        auto lrhs = lhs.args().at(1);
        if (llhs.contains(sym)) {
          return isolate(std::make_pair(llhs, rhs / lrhs), sym);
        }
        return isolate(std::make_pair(lrhs, rhs / llhs), sym);
      }
      case Op::divide: {
        auto llhs = lhs.args().at(0);
        auto lrhs = lhs.args().at(1);
        if (llhs.contains(sym)) {
          return isolate(std::make_pair(llhs, rhs * lrhs), sym);
        }
        return isolate(std::make_pair(lrhs, rhs * llhs), sym);
      }
      case Op::negate:
        return isolate(std::make_pair(lhs.args().at(0), -rhs), sym);
      case Op::reciprocal:
        return isolate(std::make_pair(lhs.args().at(0), Expr(1) / rhs), sym);
      default:
        ASSERT(0) << "cannot isolate through " << lhs.dump();
    }
  }
  ASSERT(0) << "error isolating for " << sym.name() << " in constraint "
            << lhs.dump() << " = " << rhs.dump();
  return std::make_pair(Expr(0), Expr(0));
}

// Hand crafted "constraint solver" that attempts to maximally
// derive knowledge of |symbol| expressions (Expr::size).
// size(sym) = max(all index constraints) || user provided value
// TODO improve robustness
// Any contribution of testing/impl would be highly appreciated :)
std::vector<Constraint> unify(std::vector<Constraint> constraints) {
  std::unordered_map<Symbol, std::unordered_set<Expr, Hash<Expr>>, Hash<Symbol>>
      index_constraints;
  std::unordered_map<Symbol, std::unordered_set<Expr, Hash<Expr>>, Hash<Symbol>>
      size_constraints;
  // replace size with symbolic placeholder
  std::unordered_map<Symbol, Symbol, Hash<Symbol>> size_sym_map;

  auto get_all_syms = [](const Expr& expr) {
    std::vector<Symbol> syms;
    expr.walk([&](const Expr& e) {
      if (e.type() == Expr::Type::symbol) {
        syms.emplace_back(e.symbol());
      }
      return e;
    });
    return syms;
  };

  // collect all indexing and size constraints and create a size(sym)->sym map
  for (const auto& c : constraints) {
    auto lhs_syms = get_all_syms(c.first);
    auto rhs_syms = get_all_syms(c.second);
    std::unordered_set<Symbol, Hash<Symbol>> syms;
    for (auto& sym : lhs_syms) {
      syms.insert(sym);
    }
    for (auto& sym : rhs_syms) {
      syms.insert(sym);
    }

    for (const auto& sym : syms) {
      if (can_isolate(c, sym)) {
        const auto& expr = isolate(c, sym).second;
        index_constraints[sym].insert(expr);
      }
    }

    for (const auto& sym : syms) {
      if (!size_sym_map.count(sym)) {
        size_sym_map[sym] = Symbol(sym.name() + "_size");
      }
    }
    if (c.first.type() == Expr::Type::function && c.first.op() == Op::size) {
      size_constraints[c.first.args().at(0).symbol()].insert(c.second);
    }
  }

  auto update_size_to_sym = [&](const Expr& expr) {
    auto new_expr = expr;
    for (auto& s : size_sym_map) {
      new_expr = new_expr.replace(Expr::size(s.first), s.second);
    }
    return new_expr;
  };

  for (auto& p : index_constraints) {
    auto orig_exprs = p.second;
    p.second.clear();
    for (auto& expr : orig_exprs) {
      index_constraints[p.first].insert(update_size_to_sym(expr));
    }
  }
  for (auto& p : size_constraints) {
    auto orig_exprs = p.second;
    p.second.clear();
    for (auto& expr : orig_exprs) {
      size_constraints[p.first].insert(update_size_to_sym(expr));
    }
  }

  // constraints with swapped exprs for size(sym) -> sym_size
  std::vector<Constraint> size_sym_constraints;

  for (auto& c : constraints) {
    auto lhs = c.first;
    auto rhs = c.second;
    for (auto& p : size_sym_map) {
      auto size_expr = Expr::size(p.first);
      lhs = lhs.replace(size_expr, p.second);
      rhs = rhs.replace(size_expr, p.second);
    }
    size_sym_constraints.emplace_back(lhs, rhs);
  }

  for (const auto& sym_iter : index_constraints) {
    const auto& sym = sym_iter.first;
    auto size_sym = size_sym_map.at(sym);
    for (const auto& c : size_sym_constraints) {
      auto lhs = c.first;
      auto rhs = c.second;
      // check if this constraint has a size expr
      if (!lhs.contains(size_sym) && !rhs.contains(size_sym)) {
        continue;
      }
      // can't isolate this constraint
      if (lhs.contains(size_sym) && rhs.contains(size_sym)) {
        continue;
      }
      auto new_c = isolate(Constraint(lhs, rhs), size_sym);
      size_constraints[sym].insert(new_c.second);
    }
    // derived sized functions
    // x = y + k -->
    //  |x| - 1 = |y| - 1 + |k| - 1
    //  |x| = |y| - 1 + |k| - 1 + 1
    for (const auto& expr : index_constraints.at(sym)) {
      auto size_expr = expr.walk([&](const Expr& e) {
        if (e.type() == Expr::Type::symbol) {
          if (size_sym_map.count(e.symbol())) {
            return Expr(size_sym_map.at(e.symbol())) - Expr(1);
          } else {
            // we've got an already swapped out sym_size
          }
        }
        return e;
      });
      size_constraints[sym].insert(size_expr + Expr(1));
    }
  }

  // now find any "value" constraints, these are user specified
  for (auto& p : size_constraints) {
    bool value_set = false;
    auto sym = p.first;
    auto size_exprs = p.second;
    for (auto& e : size_exprs) {
      if (e.type() == Expr::Type::value) {
        size_constraints[sym].clear();
        size_constraints[sym].insert(e);
        ASSERT(!value_set) << "size of " << sym.name()
                           << " set multiple times to different values";
        value_set = true;
      }
    }
  }

  auto sized = [&](Symbol sym) {
    const auto& exprs = size_constraints.at(sym);
    if (exprs.size() != 1) {
      return false;
    }
    const auto& expr = *exprs.begin();
    if (expr.type() == Expr::Type::value) {
      return true;
    }
    return false;
  };

  auto sized_syms = [&]() {
    std::vector<std::pair<Symbol, Expr>> sizes;
    for (auto& s : size_constraints) {
      auto sym = s.first;
      if (sized(sym)) {
        auto expr = *size_constraints.at(sym).begin();
        sizes.emplace_back(sym, expr.simplify());
      }
    }
    return sizes;
  };

  auto simply_all_sizes = [&]() {
    for (auto& p : size_constraints) {
      auto exprs = p.second;
      p.second.clear();
      for (const auto& expr_ : exprs) {
        auto expr = expr_;
        for (auto& s : sized_syms()) {
          expr = expr.replace(size_sym_map.at(s.first), s.second);
        }
        p.second.insert(expr.simplify());
      }
    }
  };

  for (auto i = 0; i < 3; ++i) {
    simply_all_sizes();
  }

  std::vector<std::pair<Expr, Expr>> output_constraints;
  for (auto& p : size_constraints) {
    if (p.second.size() == 1) {
      output_constraints.emplace_back(Expr::size(p.first),
                                      p.second.begin()->simplify());
      continue;
    }
    auto max_expr = *p.second.begin();
    for (auto& expr : p.second) {
      max_expr = Expr::max(max_expr, expr);
    }
    output_constraints.emplace_back(Expr::size(p.first), max_expr.simplify());
  }
  for (auto& p : index_constraints) {
    auto sym = p.first;
    for (auto expr : p.second) {
      output_constraints.emplace_back(sym, expr.simplify());
    }
  }

  // finally remap sym_size back into size(sym)
  for (auto& p : size_constraints) {
    auto exprs = p.second;
    p.second.clear();
    for (const auto& expr_ : exprs) {
      auto expr = expr_;
      for (auto& s : size_sym_map) {
        expr = expr.replace(s.second, Expr::size(s.first));
      }
      p.second.insert(expr);
    }
  }

  return output_constraints;
}

Expr differentiate(Expr e, Symbol sym) {
  if (!e.contains(sym)) {
    return Expr(0);
  }

  if (e == Expr(sym)) {
    return Expr(1);
  }

  if (e.type() == Expr::Type::function) {
    if (e.args().size() == 2) {
      const auto& a = e.args().at(0);
      const auto& b = e.args().at(1);
      if (e.op() == Op::add) {
        if (a.contains(sym) && !b.contains(sym)) {
          return differentiate(a, sym);
        } else if (b.contains(sym) && !a.contains(sym)) {
          return differentiate(b, sym);
        } else {
          ASSERT(a.contains(sym) && b.contains(sym));
          return differentiate(a, sym) + differentiate(b, sym);
        }
      } else if (e.op() == Op::multiply) {
        if (a.contains(sym) && !b.contains(sym)) {
          return differentiate(a, sym) * b;
        } else if (b.contains(sym) && !a.contains(sym)) {
          return differentiate(b, sym) * a;
        } else {
          ASSERT(a.contains(sym) && b.contains(sym));
          return differentiate(a, sym) * b + differentiate(b, sym) * a;
        }
      } else if (e.op() == Op::divide) {
        if (a.contains(sym) && !b.contains(sym)) {
          return differentiate(a, sym) / b;
        } else if (b.contains(sym) && !a.contains(sym)) {
          return a * differentiate(b, sym) / (b * b);
        } else {
          ASSERT(a.contains(sym) && b.contains(sym));
          return (differentiate(a, sym) * b - a * differentiate(b, sym)) /
                 (b * b);
        }
      }
    } else if (e.args().size() == 1) {
      const auto& arg = e.args().at(0);
      if (e.op() == Op::negate) {
        return -differentiate(arg, sym);
      } else if (e.op() == Op::reciprocal) {
        return differentiate(arg, sym) / (arg * arg);
      }
    }
  }

  ASSERT(0) << "Cannot differentiate " << e.dump() << " with respect to "
            << sym.name();
  return Expr(0);
}

}  // namespace symbolic
}  // namespace loop_tool
