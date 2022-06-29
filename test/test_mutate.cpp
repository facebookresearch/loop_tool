/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>
#include <loop_tool/mutate.h>

#include "test_utils.h"

TEST(MutateSplit) {
  namespace lz = ::loop_tool::lazy;
  auto mm = [](lz::Tensor A, lz::Tensor B) {
    auto M = lz::Symbol("m"), N = lz::Symbol("n"), K = lz::Symbol("k");
    auto C = A.as(M, K) * B.as(K, N);
    return C.sum(K);
  };

  lz::Tensor A(16, 16);
  lz::Tensor B(16, 17);
  auto C = mm(A, B);
  auto lt = C.loop_tree();
  std::cerr << "presplit:\n";
  std::cerr << lt.dump();
  std::cerr << '\n';
  auto r = lt.children(lt.children(lt.roots.at(0)).at(0)).at(0);
  lt = split(lt, r, 10);
  std::cerr << '\n';
  std::cerr << lt.dump();
}

TEST(MutateMerge) {
  namespace lz = ::loop_tool::lazy;
  auto N = lz::Symbol("n");
  lz::Tensor A(16);
  lz::Tensor B(16);
  auto C = A + B;
  auto lt = C.loop_tree();
  std::cerr << "presplit:\n";
  std::cerr << lt.dump();
  std::cerr << '\n';
  auto r = lt.children(lt.roots.at(0)).at(0);
  lt = split(lt, r, 10);
  std::cerr << '\n';
  std::cerr << lt.dump();
  auto c = lt.children(lt.children(lt.roots.at(0)).at(0)).at(0);
  lt = merge(lt, c);
  std::cerr << '\n';
  std::cerr << "postmerge:\n";
  std::cerr << lt.dump();
}

TEST(MutateSwap) {
  namespace lz = ::loop_tool::lazy;
  auto mm = [](lz::Tensor A, lz::Tensor B) {
    auto M = lz::Symbol("m"), N = lz::Symbol("n"), K = lz::Symbol("k");
    auto C = A.as(M, K) * B.as(K, N);
    return C.sum(K);
  };

  lz::Tensor A(16, 16);
  lz::Tensor B(16, 17);
  auto C = mm(A, B);
  auto lt = C.loop_tree();
  std::cerr << '\n';
  std::cerr << lt.dump();
  std::cerr << '\n';
  auto r = lt.children(lt.roots.at(0)).at(0);
  lt = split(lt, r, 10);

  auto a = lt.children(lt.children(lt.roots.at(0)).at(0)).at(0);
  auto b = lt.children(a).at(0);
  lt = swap_loops(lt, a, b);

  std::cerr << '\n';
  std::cerr << lt.dump();

  C.compile();
  C.set(lt);
  std::cerr << C.code();
}

TEST(MutateSubTree) {
  namespace lz = ::loop_tool::lazy;
  auto mm = [](lz::Tensor A, lz::Tensor B) {
    auto M = lz::Symbol("m"), N = lz::Symbol("n"), K = lz::Symbol("k");
    auto C = A.as(M, K) * B.as(K, N);
    return C.sum(K);
  };

  lz::Tensor A(16, 16);
  lz::Tensor B(16, 16);
  lz::Tensor C(16, 16);
  lz::Tensor D(16, 16);
  lz::Tensor E(16, 16);
  auto F = mm(A, B);
  auto G = mm(C, D);
  auto H = mm(E, F);
  auto I = mm(G, H);
  auto lt = I.loop_tree();
  std::cerr << "old loop_tree" << lt.dump() << "\n";
  lt = subtree(lt, lt.roots[1]);
  std::cerr << "new loop_tree: " << lt.dump() << "\n";
}

TEST(MutateGenFeature) {
  namespace lz = ::loop_tool::lazy;
  auto mm = [](lz::Tensor A, lz::Tensor B) {
    auto M = lz::Symbol("m"), N = lz::Symbol("n"), K = lz::Symbol("k");
    auto C = A.as(M, K) * B.as(K, N);
    return C.sum(K);
  };

  lz::Tensor A(16, 16);
  lz::Tensor B(16, 16);
  auto C = mm(A, B);
  gen_feature(C.ir());
}