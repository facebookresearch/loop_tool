/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/symbolic.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "loop_tool/error.h"

namespace loop_tool {
namespace symbolic {

bool associative(Op op) {
  switch (op) {
    case Op::add:
    case Op::multiply:
      return true;
    default:
      return false;
  }
  return false;
}

const int32_t Symbol::getNewId() {
  static int32_t symbol_count_ = 0;
  return symbol_count_++;
}

const int32_t Symbol::id() const { return id_; }
size_t Symbol::hash() const { return symbolic::hash(id_); }
bool Symbol::operator==(const Symbol& s) const { return s.id() == id_; }
bool Symbol::operator!=(const Symbol& s) const { return !(s.id() == id_); }
std::string Symbol::name() const { return name_; }

Symbol::operator Expr() const { return Expr(*this); }
Expr Symbol::operator+(const Symbol& rhs) const {
  return Expr(*this) + Expr(rhs);
}
Expr Symbol::operator*(const Symbol& rhs) const {
  return Expr(*this) * Expr(rhs);
}

Expr Symbol::operator+(const Expr& rhs) const { return Expr(*this) + rhs; }
Expr Symbol::operator*(const Expr& rhs) const { return Expr(*this) * rhs; }

ExprImpl::ExprImpl(Op op, const Expr& e, bool simplified)
    : type_(Type::function), op_(op), simplified_(simplified) {
  args_.emplace_back(e.impl_);
  init();
}

ExprImpl::ExprImpl(Op op, const Expr& a, const Expr& b, bool simplified)
    : type_(Type::function), op_(op), simplified_(simplified) {
  if (!associative(op) || (a.hash() > b.hash())) {
    args_.emplace_back(a.impl_);
    args_.emplace_back(b.impl_);
  } else {
    args_.emplace_back(b.impl_);
    args_.emplace_back(a.impl_);
  }
  init();
}

void ExprImpl::init() {
  hash_ = symbolic::hash((int)op_);
  if (type_ == Type::value) {
    hash_ = symbolic::hash_combine(hash_, symbolic::hash(val_));
  } else if (type_ == Type::symbol) {
    hash_ = symbolic::hash_combine(hash_, symbolic::hash(1337));
    symbol_hash_ = symbolic::hash_combine(hash_, symbol_.hash());
  }
  for (const auto& expr : args_) {
    hash_ = symbolic::hash_combine(hash_, expr->hash(false));
    symbol_hash_ = symbolic::hash_combine(symbol_hash_, expr->hash(true));
  }
}

Expr Expr::walk(std::function<Expr(const Expr&)> f) const {
  if (type() == Type::function) {
    const auto& args_ = impl_->args_;
    if (args_.size() == 2) {
      const auto& a = Expr(args_[0]).walk(f);
      const auto& b = Expr(args_[1]).walk(f);
      if (a.impl_.get() != args_[0].get() || b.impl_.get() != args_[1].get()) {
        return f(Expr(op(), a, b));
      }
    } else if (args_.size() == 1) {
      const auto& a = Expr(args_[0]).walk(f);
      if (a.impl_.get() != args_[0].get()) {
        return f(Expr(op(), a));
      }
    }
  }
  return f(*this);
}

void Expr::visit(std::function<void(const Expr&)> f) const {
  if (type() == Type::function) {
    for (const auto& arg : impl_args()) {
      Expr(arg).visit(f);
    }
  }
  f(*this);
}

std::string Expr::dump(
    bool short_form,
    const std::unordered_map<Symbol, std::string, Hash<Symbol>>& replacements)
    const {
  std::stringstream ss;
  if (type() == Type::value) {
    ss << value();
  } else if (type() == Type::symbol) {
    auto sym = symbol();
    if (replacements.count(sym)) {
      ss << replacements.at(sym);
    } else {
      ss << symbol().name();
      if (!short_form) {
        ss << "[id:" << symbol().id() << "]";
      }
    }
  } else if (op() == Op::size) {
    ASSERT(impl_args().size() == 1)
        << "invalid size function found: " << impl_args().size()
        << " arguments";
    ss << "|" << Expr(impl_args().at(0)).dump(short_form, replacements) << "|";
  } else if (op() == Op::max) {
    ASSERT(impl_args().size() == 2);
    ss << "max(" << Expr(impl_args().at(0)).dump(short_form, replacements)
       << ", " << Expr(impl_args().at(1)).dump(short_form, replacements) << ")";
  } else if (op() == Op::negate) {
    ASSERT(impl_args().size() == 1);
    auto arg_ = arg(0);
    ss << "-";
    if (arg_.type() == Type::function) {
      ss << "(" << Expr(impl_args().at(0)).dump(short_form, replacements)
         << ")";
    } else {
      ss << Expr(impl_args().at(0)).dump(short_form, replacements);
    }
  } else if (op() == Op::reciprocal) {
    ASSERT(impl_args().size() == 1);
    ss << "(1 / " << Expr(impl_args().at(0)).dump(short_form, replacements)
       << ")";
  } else {
    ASSERT(impl_args().size() == 2);
    auto lhs = Expr(impl_args().at(0));
    auto rhs = Expr(impl_args().at(1));
    if (lhs.op() == Op::constant || lhs.impl_args().size() == 1) {
      ss << lhs.dump(short_form, replacements);
    } else {
      ss << "(" << lhs.dump(short_form, replacements) << ")";
    }
    // we pretty print addition of negatives
    if (op() == Op::add) {
      ss << "+";
    } else if (op() == Op::multiply) {
      ss << "*";
    } else if (op() == Op::divide) {
      ss << "/";
    } else if (op() == Op::modulo) {
      ss << "%";
    } else {
      ASSERT(0) << "can't print this op id " << (int)op();
    }

    if (rhs.op() == Op::constant || rhs.impl_args().size() == 1) {
      ss << rhs.dump(short_form, replacements);
    } else {
      ss << "(" << rhs.dump(short_form, replacements) << ")";
    }
  }
  return ss.str();
}

std::vector<Symbol> Expr::symbols(bool include_sized) const {
  std::vector<Symbol> out;
  std::unordered_set<int32_t> seen;
  auto size_removed_expr = [&]() {
    if (!include_sized) {
      return this->walk([&](const Expr& e) {
        if (e.op() == Op::size) {
          return Expr(0);
        }
        return e;
      });
    }
    return *this;
  }();
  size_removed_expr.visit([&](const Expr& e) {
    if (e.type() == Type::symbol) {
      const auto& sym = e.symbol();
      if (seen.count(sym.id())) {
        return e;
      }
      seen.insert(sym.id());
      out.emplace_back(sym);
    }
    return e;
  });
  return out;
}

bool Expr::operator!=(const Expr& rhs) const { return !(*this == rhs); }

bool Expr::operator==(const Expr& rhs) const {
  if (hash(true) != rhs.hash(true)) {
    return false;
  }
  if (impl_.get() == rhs.impl_.get()) {
    return true;
  }
  if (type() == Type::value) {
    return rhs.type() == Type::value && rhs.value() == value();
  } else if (type() == Type::symbol) {
    return rhs.type() == Type::symbol && rhs.symbol() == symbol();
  }
  if (rhs.type() != Type::function) {
    return false;
  }
  bool match = true;
  if (impl_args().size() == rhs.impl_args().size()) {
    auto lhs_args = impl_args();
    auto rhs_args = rhs.impl_args();
    for (auto i = 0; i < lhs_args.size(); ++i) {
      match &= Expr(lhs_args.at(i)) == Expr(rhs_args.at(i));
    }
  } else {
    match = false;
  }
  return rhs.op() == op() && match;
}

Expr Expr::replace(const Expr& target_e, Symbol B) const {
  return walk([&](const Expr& e) {
    if (target_e == e) {
      return Expr(B);
    }
    return e;
  });
}

Expr Expr::replace(const Expr& target_e, int64_t c) const {
  return walk([&](const Expr& e) {
    if (target_e == e) {
      return Expr(c);
    }
    return e;
  });
}

Expr Expr::replace(Symbol A, Expr new_e) const {
  return walk([&](const Expr& e) {
    if (e.type() == Type::symbol && e.symbol() == A) {
      return new_e;
    }
    return e;
  });
}

Expr Expr::replace(Symbol A, Symbol B) const {
  return walk([&](const Expr& e) {
    if (e.type() == Type::symbol && e.symbol() == A) {
      return Expr(B);
    }
    return e;
  });
}

Expr Expr::replace(Symbol A, int64_t c) const {
  return walk([&](const Expr& e) {
    if (e.type() == Type::symbol && e.symbol() == A) {
      return Expr(c);
    }
    return e;
  });
}

bool Expr::can_evaluate() const {
  bool can = true;
  if (type() == Expr::Type::function && op() == Op::max) {
    auto lhs = Expr(impl_args().at(0));
    auto rhs = Expr(impl_args().at(1));
    const auto lhs_can = lhs.can_evaluate();
    const auto rhs_can = rhs.can_evaluate();
    return lhs_can || rhs_can;
  }
  visit([&](const Expr& e) {
    if (e.type() == Expr::Type::symbol) {
      can = false;
    }
  });
  return can;
}

float Expr::evaluate() const {
  ASSERT(can_evaluate());
  if (type() != Expr::Type::function) {
    ASSERT(type() == Expr::Type::value) << "can't evaluate " << dump();
    return (float)value();
  }
  switch (op()) {
    case Op::add: {
      auto lhs = Expr(impl_args().at(0)).evaluate();
      auto rhs = Expr(impl_args().at(1)).evaluate();
      return lhs + rhs;
    }
    case Op::multiply: {
      auto lhs = Expr(impl_args().at(0)).evaluate();
      auto rhs = Expr(impl_args().at(1)).evaluate();
      return lhs * rhs;
    }
    case Op::divide: {
      auto lhs = Expr(impl_args().at(0)).evaluate();
      auto rhs = Expr(impl_args().at(1)).evaluate();
      return lhs / rhs;
    }
    case Op::modulo: {
      auto lhs = Expr(impl_args().at(0)).evaluate();
      auto rhs = Expr(impl_args().at(1)).evaluate();
      std::cerr << "WARNING: evaluating modular arithmetic";
      return ((int64_t)lhs % int64_t(rhs));
    }
    case Op::max: {
      auto lhs = Expr(impl_args().at(0));
      auto rhs = Expr(impl_args().at(1));
      const auto lhs_can = lhs.can_evaluate();
      const auto rhs_can = rhs.can_evaluate();
      if (lhs_can && !rhs_can) {
        return lhs.evaluate();
      } else if (!lhs_can && rhs_can) {
        return rhs.evaluate();
      }
      return std::max(lhs.evaluate(), rhs.evaluate());
    }
    case Op::negate: {
      return -Expr(impl_args().at(0)).evaluate();
    }
    default:
      ASSERT(0) << "couldn't evaluate expression " << dump();
      return 0;
  }
  return 0;
}

Expr Expr::simplify() const {
  if (simplified()) {
    return *this;
  }
  auto get_pair = [&]() {
    ASSERT(impl_args().size() == 2);
    return std::make_pair(arg(0).simplify(), arg(1).simplify());
  };
  switch (op()) {
    case Op::add: {
      const auto& pair = get_pair();
      const auto& lhs = pair.first;
      const auto& rhs = pair.second;
      if (lhs.type() == Expr::Type::value) {
        if (rhs.type() == Expr::Type::value) {
          return Expr(lhs.value() + rhs.value());
        }
        if (lhs.value() == 0) {
          return rhs;
        }
        if (rhs.op() == Op::add) {
          auto rlhs = rhs.arg(0);
          auto rrhs = rhs.arg(1);
          if (rlhs.type() == Expr::Type::value) {
            return (Expr(rlhs.value() + lhs.value()) + rrhs).simplify();
          }
        }
      }
      if (rhs.type() == Expr::Type::value) {
        if (rhs.value() == 0) {
          return lhs;
        }
      }
      return Expr(op(), lhs, rhs, true);
    }
    case Op::multiply: {
      const auto& pair = get_pair();
      const auto& lhs = pair.first;
      const auto& rhs = pair.second;
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
      return Expr(op(), lhs, rhs, true);
    }
    case Op::divide: {
      const auto& pair = get_pair();
      const auto& lhs = pair.first;
      const auto& rhs = pair.second;
      if (lhs.type() == Expr::Type::value) {
        if (rhs.type() == Expr::Type::value) {
          if (rhs.value() && lhs.value() % rhs.value() == 0) {
            return Expr(lhs.value() / rhs.value());
          }
        }
      }
      if (rhs.type() == Expr::Type::value && rhs.value() == 1) {
        return lhs;
      }
      return Expr(op(), lhs, rhs, true);
    }
    case Op::modulo: {
      const auto& pair = get_pair();
      const auto& lhs = pair.first;
      const auto& rhs = pair.second;
      if (lhs.type() == Expr::Type::value) {
        if (rhs.type() == Expr::Type::value) {
          if (rhs.value() && lhs.value() % rhs.value() == 0) {
            return Expr(lhs.value() % rhs.value());
          }
        }
      }
      if (rhs.type() == Expr::Type::value && rhs.value() == 1) {
        return Expr(0);
      }
      return Expr(op(), lhs, rhs, true);
    }
    case Op::max: {
      const auto& pair = get_pair();
      const auto& lhs = pair.first;
      const auto& rhs = pair.second;
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
      return Expr(op(), lhs, rhs, true);
    }
    case Op::negate: {
      const auto& arg_ = arg(0).simplify();
      if (arg_.type() == Expr::Type::value) {
        return Expr(-arg_.value());
      }
      if (arg_.type() == Expr::Type::function && arg_.op() == Op::negate) {
        return arg_.arg(0).simplify();
      }
      if (arg_.type() == Expr::Type::function && arg_.op() == Op::add) {
        return (-arg_.arg(0) - arg_.arg(1)).simplify();
      }
      return Expr(op(), arg_, true);
    }
    case Op::size: {
      const auto& arg_ = arg(0).simplify();
      if (arg_.type() == Expr::Type::value) {
        return Expr(arg_.value());
      }
      return Expr(op(), arg_, true);
    }
    default: {
      if (impl_args().size() == 2) {
        const auto& pair = get_pair();
        return Expr(op(), pair.first, pair.second);
      } else if (impl_args().size() == 1) {
        return Expr(op(), arg(0).simplify());
      }
      return *this;
    }
  };
  ASSERT(0);
  return *this;
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
      auto args_size = e.impl_args().size();
      for (auto i = 0; i < args_size; ++i) {
        res &= can_isolate(e.arg(i), sym);
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
  if (lhs.contains(sym) + rhs.contains(sym) != 1) {
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
        << "cannot isolate " << sym.name() << " through " << lhs.dump()
        << ", you may need to update the can_isolate function";
    switch (lhs.op()) {
      case Op::add: {
        auto llhs = lhs.arg(0);
        auto lrhs = lhs.arg(1);
        if (llhs.contains(sym)) {
          return isolate(std::make_pair(llhs, rhs - lrhs), sym);
        }
        return isolate(std::make_pair(lrhs, rhs - llhs), sym);
      }
      case Op::multiply: {
        auto llhs = lhs.arg(0);
        auto lrhs = lhs.arg(1);
        if (llhs.contains(sym)) {
          return isolate(std::make_pair(llhs, rhs / lrhs), sym);
        }
        ASSERT(lrhs.contains(sym));
        return isolate(std::make_pair(lrhs, rhs / llhs), sym);
      }
      case Op::divide: {
        auto llhs = lhs.arg(0);
        auto lrhs = lhs.arg(1);
        if (llhs.contains(sym)) {
          return isolate(std::make_pair(llhs, rhs * lrhs), sym);
        }
        return isolate(std::make_pair(lrhs, rhs * llhs), sym);
      }
      case Op::negate:
        return isolate(std::make_pair(lhs.arg(0), -rhs), sym);
      case Op::reciprocal:
        return isolate(std::make_pair(lhs.arg(0), Expr(1) / rhs), sym);
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
std::vector<Constraint> unify(std::vector<Constraint> constraints_) {
  // 1. Get all symbols
  std::unordered_set<Symbol, Hash<Symbol>> all_symbols;

  auto get_all_syms = [](const Expr& expr) {
    std::vector<Symbol> syms;
    expr.visit([&](const Expr& e) {
      if (e.type() == Expr::Type::symbol) {
        syms.emplace_back(e.symbol());
      }
    });
    return syms;
  };

  for (const auto& c : constraints_) {
    for (auto& sym : get_all_syms(c.first)) {
      all_symbols.insert(sym);
    }
    for (auto& sym : get_all_syms(c.second)) {
      all_symbols.insert(sym);
    }
  }

  // 2. Symbolicate all size expressions (map symbol -> Size(symbol))
  std::unordered_map<Symbol, Symbol, Hash<Symbol>> size_sym_map;

  for (const auto& sym : all_symbols) {
    size_sym_map[sym] = Symbol(sym.name() + "_size");
  }

  // 3. Remap size expressions in the constraint list
  std::vector<Constraint> constraints;
  std::unordered_set<int64_t> seen_hashes;

  for (const auto& c : constraints_) {
    auto lhs = c.first;
    auto rhs = c.second;
    for (auto& p : size_sym_map) {
      auto size_expr = Expr::size(p.first);
      lhs = lhs.replace(size_expr, p.second);
      rhs = rhs.replace(size_expr, p.second);
    }
    auto h = symbolic::hash_combine(lhs.hash(true), rhs.hash(true));
    if (seen_hashes.count(h)) {
      continue;
    }
    seen_hashes.insert(h);
    constraints.emplace_back(lhs, rhs);
  }

  // 4. Collect size-only constraints (no symbols)
  std::unordered_map<Symbol, std::unordered_set<Expr, Hash<Expr>>, Hash<Symbol>>
      size_constraints;

  for (const auto& c : constraints) {
    bool size_only = true;
    for (const auto& sym : all_symbols) {
      if (c.first.contains(sym) || c.second.contains(sym)) {
        size_only = false;
        break;
      }
    }
    if (!size_only) {
      continue;
    }
    for (const auto& p : size_sym_map) {
      const auto& sym = p.first;
      const auto& size_sym = p.second;
      if (!can_isolate(c, size_sym)) {
        continue;
      }
      const auto& expr = isolate(c, size_sym).second;
      size_constraints[sym].insert(expr);
    }
  }

  for (auto& p : size_constraints) {
    if (p.second.size() > 1) {
      const auto& exprs = p.second;
      for (const auto& expr : exprs) {
        if (expr.type() == Expr::Type::value) {
          size_constraints[p.first] = {expr};
          break;
        }
      }
    }
  }

  // 5. Simplify the size constraints
  auto sized = [&](Symbol sym) {
    if (!size_constraints.count(sym)) {
      return false;
    }
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
    for (const auto& s : size_constraints) {
      auto sym = s.first;
      if (sized(sym)) {
        auto expr = *s.second.begin();
        sizes.emplace_back(sym, expr.simplify());
      }
    }
    return sizes;
  };

  auto simplify_all_sizes = [&]() -> bool {
    bool changed = false;
    for (auto& p : size_constraints) {
      auto exprs = p.second;
      p.second.clear();
      for (const auto& expr_ : exprs) {
        auto expr = expr_;
        for (const auto& s : sized_syms()) {
          expr = expr.replace(size_sym_map.at(s.first), s.second);
        }
        p.second.insert(expr.simplify());
      }
      if (exprs.size() == p.second.size()) {
        for (const auto& expr : exprs) {
          if (!p.second.count(expr)) {
            changed = true;
          }
        }
      } else {
        changed = true;
      }
    }
    return changed;
  };

  auto resolve_values = [&]() {
    for (auto& p : size_constraints) {
      bool value_set = false;
      auto sym = p.first;
      auto size_exprs = p.second;
      for (auto& e : size_exprs) {
        if (e.type() == Expr::Type::value) {
          ASSERT(!value_set)
              << "size of " << sym.name()
              << " set multiple times to different values"
              << " (new value: " << e.dump() << " old:"
              << " " << size_constraints[sym].begin()->dump() << ")";
          size_constraints[sym].clear();
          size_constraints[sym].insert(e);
          value_set = true;
        }
      }
    }
  };

  {
    int limit = 1000;
    while (simplify_all_sizes() && (limit--) > 0) {
      resolve_values();
    }
  }

  // 6. Derive all indexing constraints
  std::unordered_map<Symbol, std::unordered_set<Expr, Hash<Expr>>, Hash<Symbol>>
      index_constraints;

  // collect all indexing and size constraints and create a size(sym)->sym map
  for (const auto& c : constraints) {
    for (const auto& sym : all_symbols) {
      if (can_isolate(c, sym)) {
        const auto& expr = isolate(c, sym).second;
        index_constraints[sym].insert(expr);
      }
    }
  }

  // 7. Derive unknown size constraints from index constraints
  auto derive_size_expressions = [&]() {
    for (const auto& sym : all_symbols) {
      if (sized(sym)) {
        continue;
      }
      if (!index_constraints.count(sym)) {
        continue;
      }
      // derived sized functions
      // x = y + k -->
      //  |x| - 1 = |y| - 1 + |k| - 1
      //  |x| = |y| - 1 + |k| - 1 + 1
      for (const auto& expr : index_constraints.at(sym)) {
        auto size_expr = expr.walk([&](const Expr& e) {
          if (e.type() == Expr::Type::symbol &&
              size_sym_map.count(e.symbol())) {
            return Expr(size_sym_map.at(e.symbol())) - Expr(1);
          }
          return e;
        });
        // if we've already had our sizes bound by previous iterations
        // of unification, we can skip this step entirely
        if (size_constraints.count(sym)) {
          continue;
        }
        size_constraints[sym].insert(size_expr + Expr(1));
      }
    }
  };

  derive_size_expressions();
  {
    int limit = 1000;
    while (simplify_all_sizes() && (limit--) > 0) {
    }
  }

  // 8. All done, take the maximum if there are multiple size constraints
  std::vector<std::pair<Expr, Expr>> output_constraints;

  auto map_to_size_expr = [&](const Expr& e) {
    auto expr = e;
    for (auto& s : size_sym_map) {
      expr = expr.replace(s.second, Expr::size(s.first));
    }
    return expr.simplify();
  };

  for (auto& p : size_constraints) {
    if (p.second.size() == 1) {
      output_constraints.emplace_back(Expr::size(p.first),
                                      map_to_size_expr(*p.second.begin()));
      continue;
    }
    auto max_expr = *p.second.begin();
    for (auto& expr : p.second) {
      max_expr = Expr::max(max_expr, expr);
    }
    output_constraints.emplace_back(Expr::size(p.first),
                                    map_to_size_expr(max_expr));
  }

  for (auto& p : index_constraints) {
    auto sym = p.first;
    std::unordered_set<Expr, Hash<Expr>> mapped_exprs;
    for (auto expr : p.second) {
      mapped_exprs.insert(map_to_size_expr(expr));
    }
    for (auto expr : mapped_exprs) {
      output_constraints.emplace_back(sym, expr);
    }
  }
  std::sort(output_constraints.begin(), output_constraints.end(),
            [](const std::pair<Expr, Expr>& a, const std::pair<Expr, Expr>& b) {
              return hash_combine(a.first.hash(), a.second.hash()) <
                     hash_combine(b.first.hash(), b.second.hash());
            });

  return output_constraints;
}

std::vector<Constraint> evaluate(
    const std::vector<Constraint>& old_constraints) {
  std::unordered_map<Symbol, int64_t, symbolic::Hash<Symbol>> evaluated_sizes;
  std::vector<Constraint> constraints;
  for (const auto& c : old_constraints) {
    constraints.emplace_back(c);
  }

  auto pass = [&]() -> bool {
    bool changed = false;
    // find evaluations
    for (const auto& c : constraints) {
      if (c.first.op() == symbolic::Op::size && c.second.can_evaluate()) {
        ASSERT(c.first.impl_args().size() == 1);
        ASSERT(Expr(c.first.impl_args().at(0)).type() == Expr::Type::symbol);
        const auto& sym = c.first.arg(0).symbol();
        if (evaluated_sizes.count(sym)) {
          continue;
        }
        evaluated_sizes[sym] = c.second.evaluate();
        changed = true;
      }
    }
    if (!changed) {
      return changed;
    }
    // replace constraints
    for (auto& c : constraints) {
      auto expr = c.second;
      for (const auto& sym : expr.symbols()) {
        if (evaluated_sizes.count(sym)) {
          expr = expr.replace(Expr::size(sym), evaluated_sizes.at(sym));
        }
      }
      c.second = expr;
    }
    return changed;
  };

  int limit = 1000;
  while (pass() && (limit--) > 0)
    ;

  return constraints;
}

Expr differentiate(Expr e, Symbol sym) {
  if (!e.contains(sym)) {
    return Expr(0);
  }

  if (e == Expr(sym)) {
    return Expr(1);
  }

  if (e.type() == Expr::Type::function) {
    if (e.impl_args().size() == 2) {
      const auto& a = e.arg(0);
      const auto& b = e.arg(1);
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
    } else if (e.impl_args().size() == 1) {
      const auto& arg = e.arg(0);
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

Expr intercept(Expr e) {
  auto symbols = e.symbols();
  for (const auto& s : symbols) {
    e = e.replace(s, Expr(0)).simplify();
  }
  return e;
}

}  // namespace symbolic
}  // namespace loop_tool
