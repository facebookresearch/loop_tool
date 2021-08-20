/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>

#include "test_utils.h"

using namespace loop_tool;
using namespace loop_tool::testing;

TEST(SymbolicBasic) {
  namespace lz = loop_tool::lazy;
  std::vector<lz::Constraint> constraints;
  lz::Symbol A("A");
  lz::Symbol B("B");
  lz::Symbol C("C");
  lz::Symbol D("D");
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(C), lz::Expr::size(B) * lz::Expr(9)));
  constraints.emplace_back(std::make_pair(lz::Expr::size(A), lz::Expr(8)));
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(B), lz::Expr::size(A) + lz::Expr(2)));
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(D), lz::Expr::size(A) + lz::Expr::size(C)));

  auto out = unify(constraints);
  for (auto p : out) {
    std::cout << p.first.dump() << " = " << p.second.dump() << "\n";
    if (p.first == lz::Expr::size(D)) {
      ASSERT(p.second.value() == 98);
    }
  }
}

TEST(SymbolicUnbound) {
  namespace lz = loop_tool::lazy;
  std::vector<lz::Constraint> constraints;
  lz::Symbol A("A");
  lz::Symbol B("B");
  lz::Symbol C("C");
  lz::Symbol D("D");
  // constraints.emplace_back(std::make_pair(A, lz::Expr(8)));
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(C), lz::Expr(B) * lz::Expr(9)));
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(B), lz::Expr(A) + lz::Expr(2)));
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(D), lz::Expr(A) + lz::Expr(C)));
  for (auto p : constraints) {
    // std::cout << p.first.name() << " = " << p.second.dump() << "\n";
  }

  auto out = unify(constraints);
  for (auto p : out) {
    // std::cout << p.first.name() << " = " << p.second.dump() << "\n";
  }
}
