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
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(C), lz::Expr(B) * lz::Expr(9)));
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(B), lz::Expr(A) + lz::Expr(2)));
  constraints.emplace_back(
      std::make_pair(lz::Expr::size(D), lz::Expr(A) + lz::Expr(C)));

  auto out = unify(constraints);
}

TEST(SymbolicDerivative) {
  namespace lz = loop_tool::lazy;
  lz::Symbol N("N"), N_o("N_o"), K("K");
  {
    auto d = loop_tool::symbolic::differentiate(N_o + K, N_o);
    ASSERT(d == lz::Expr(1));
  }
  {
    auto d = loop_tool::symbolic::differentiate(N_o + K, N);
    ASSERT(d == lz::Expr(0));
  }
  {
    auto d = loop_tool::symbolic::differentiate(lz::Expr(2) * N_o + K, N_o);
    ASSERT(d == lz::Expr(2));
  }
  {
    auto d =
        loop_tool::symbolic::differentiate(lz::Expr(2) * N_o + K * N_o, N_o);
    ASSERT(d == lz::Expr(2) + K) << "found " << d.dump();
  }
  {
    auto d = loop_tool::symbolic::differentiate(N + lz::Expr(2) * N_o + K * N_o,
                                                N_o);
    ASSERT(d == lz::Expr(2) + K) << "found " << d.dump();
  }
  {
    auto d = loop_tool::symbolic::differentiate(
        N * (lz::Expr(2) * N_o + K * N_o) + N_o, N_o);
    ASSERT(d == N * (lz::Expr(2) + K) + lz::Expr(1)) << "found " << d.dump();
  }
}

TEST(SymbolicPaddedConv) {
  namespace lz = loop_tool::lazy;
  auto xi = lz::Symbol("xi");
  auto xp = lz::Symbol("xp");
  auto xo = lz::Symbol("xo");
  auto k = lz::Symbol("k");
  std::vector<lz::Constraint> constraints = {
      {xi + lz::Expr(k) / lz::Expr(2), xp},  // pad left
      {lz::Expr::size(xp),
       lz::Expr::size(xi) + lz::Expr::size(k) / lz::Expr(2)},  // pad right
      {xp, xo + k},                                            // conv
      {lz::Expr::size(xo), lz::Expr(100)},
      {lz::Expr::size(k), lz::Expr(3)},
      //{lz::Expr::size(xp), lz::Expr(102)},
      {lz::Expr::size(xi), lz::Expr(100)}};
  auto out = unify(constraints);
}

TEST(SymbolicConcat) {
  namespace lz = loop_tool::lazy;
  auto k = lz::Symbol("k");
  auto j = lz::Symbol("j");
  auto kj = lz::Symbol("kj");
  std::vector<lz::Constraint> constraints = {
      {kj, k},
      {kj, j + lz::Expr::size(k)},
      {lz::Expr::size(k), lz::Expr(10)},
      {lz::Expr::size(j), lz::Expr(7)},
  };
  auto out = unify(constraints);
  for (auto& p : out) {
    if (p.first == lz::Expr::size(kj)) {
      ASSERT(p.second == lz::Expr(17))
          << "found kj size to be " << p.second.dump();
    }
    std::cerr << p.first.dump() << ": " << p.second.dump() << "\n";
  }
}

TEST(SymbolicCanonicalization) {}
