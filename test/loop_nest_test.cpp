/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>

#include "test_utils.h"

using namespace loop_tool::testing;

TEST(LoopNestBackend) {
  loop_tool::ScopedBackend sb("loop_nest");
  namespace lz = ::loop_tool::lazy;
  auto mm = [](lz::Tensor A, lz::Tensor B) {
    auto M = lz::Symbol("M");
    auto N = lz::Symbol("N");
    auto K = lz::Symbol("K");
    auto C = A.as(M, K) * B.as(K, N);
    return C.sum(K);
  };

  auto M = 16;
  auto N = 16;
  auto K = 16;

  lz::Tensor A(M, K);
  lz::Tensor B(K, N);
  for (auto i = 0; i < M * K; ++i) {
    A.data<float>()[i] = 1;
    B.data<float>()[i] = 2;
  }
  auto C = mm(A, B);
  auto d = C.data<float>();
  for (auto i = 0; i < M * N; ++i) {
    std::cerr << d[i] << " ";
  }
  std::cerr << "\n";
  C.clear_cache();
}

TEST(LoopNestMM) {
  loop_tool::ScopedBackend sb("loop_nest");
  namespace lz = ::loop_tool::lazy;
  auto mm = [](lz::Tensor A, lz::Tensor B) {
    auto M = lz::Symbol("M");
    auto N = lz::Symbol("N");
    auto K = lz::Symbol("K");
    auto C = A.as(M, K) * B.as(K, N);
    return C.sum(K);
  };

  auto M = 16;
  auto N = 16;
  auto K = 16;

  lz::Tensor A(M, K);
  lz::Tensor B(K, N);
  rand(A.data<float>(), M * K);
  rand(B.data<float>(), K * N);

  auto C = mm(A, B);
  lz::Tensor C_ref(M * N);
  ref_mm(A.data<float>(), B.data<float>(), M, N, K, C_ref.data<float>());

  ASSERT(all_close(C_ref.data<float>(), C.data<float>(), M * N));
  C.clear_cache();
}

TEST(LoopNestConv) {
  loop_tool::ScopedBackend sb("loop_nest");
  namespace lz = ::loop_tool::lazy;

  auto conv = [](lz::Tensor X, lz::Tensor w) {
    lz::Symbol N("N"), M("M"), C("C"), H("H"), Ho("Ho"), W("W"), Wo("Wo"),
        Kh("Kh"), Kw("Kw");
    X = X.as(N, C, H, W);
    w = w.as(M, C, Kh, Kw);
    auto X_im2col = X.to({N, C, Ho, Kh, Wo, Kw}, lz::Constraint(H, Ho + Kh),
                         lz::Constraint(W, Wo + Kw));
    auto Y = (X_im2col * w).sum(Kh, Kw, C);
    return Y.transpose({N, M, Ho, Wo});
  };

  auto N = 4;
  auto M = 64;
  auto C = 64;
  auto HW = 8;
  auto K = 3;
  auto HWo = HW - K + 1;

  lz::Tensor A(N, C, HW, HW);
  lz::Tensor B(M, C, K, K);
  rand(A.data<float>(), A.numel());
  rand(B.data<float>(), B.numel());

  auto C_lt = conv(A, B);
  std::cerr << C_lt.numel() << " vs " << (N * M * HWo * HWo) << "\n";
  ASSERT(C_lt.numel() == N * M * HWo * HWo);
  lz::Tensor C_ref(C_lt.numel());
  ref_conv(A.data<float>(), B.data<float>(), N, M, C, HW, K,
           C_ref.data<float>());

  ASSERT(all_close(C_ref.data<float>(), C_lt.data<float>(), C_lt.numel()));
}

TEST(LoopNestEmbedded) {
  loop_tool::ScopedBackend sb("cpu_interpreted");
  namespace lz = ::loop_tool::lazy;
  auto mm = [](lz::Tensor A, lz::Tensor B) {
    auto M = lz::Symbol("M");
    auto N = lz::Symbol("N");
    auto K = lz::Symbol("K");
    auto C = A.as(M, K) * B.as(K, N);
    return C.sum(K);
  };

  auto M = 16;
  auto N = 16;
  auto K = 16;

  lz::Tensor A(M, K);
  lz::Tensor B(K, N);
  rand(A.data<float>(), M * K);
  rand(B.data<float>(), K * N);

  auto C = mm(A, B);
  auto tree = C.loop_tree();
  tree = annotate(tree, tree.roots[0], "[loop_nest]");
  C.set(tree);
  std::cerr << "TRE IS " << tree.dump() << "\n";
  lz::Tensor C_ref(M * N);
  ref_mm(A.data<float>(), B.data<float>(), M, N, K, C_ref.data<float>());

  ASSERT(all_close(C_ref.data<float>(), C.data<float>(), M * N));
  C.clear_cache();
}
