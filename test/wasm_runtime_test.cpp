/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>

#include "test_utils.h"

using namespace loop_tool::testing;

TEST(WasmBackend) {
  loop_tool::ScopedBackend sb("wasm");
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

TEST(WasmMM) {
  loop_tool::ScopedBackend sb("wasm");
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

TEST(WasmConv) {
  loop_tool::ScopedBackend sb("wasm");
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

TEST(WasmConcat1D) {
  loop_tool::ScopedBackend sb("wasm");
  namespace lz = ::loop_tool::lazy;
  lz::Symbol N("N"), M("M"), NM("NM");
  lz::Tensor A(N);
  lz::Tensor B(M);
  A.bind(nullptr, {8});
  B.bind(nullptr, {5});
  auto A_ = A.to({NM}, lz::Constraint(NM, N),
                 lz::Constraint(lz::Expr::size(NM),
                                lz::Expr::size(N) + lz::Expr::size(M)));
  auto B_ = B.to({NM}, lz::Constraint(NM, M + lz::Expr::size(N)));
  auto C = A_ + B_;
  std::cerr << C.loop_tree().dump() << "\n";
  std::cerr << C.code() << "\n";
  std::cerr << "shape:\n";
  for (auto s : C.shape()) {
    std::cerr << s.name() << "\n";
  }
  ASSERT(C.size(0) == 13) << "size is " << C.size(0);
  for (auto i = 0; i < 8; ++i) {
    A.data<float>()[i] = i;
  }
  for (auto i = 0; i < 5; ++i) {
    B.data<float>()[i] = i;
  }
  for (auto i = 0; i < 13; ++i) {
    std::cerr << "C[" << i << "]: " << C.data<float>()[i] << "\n";
  }
  // ASSERT(C.data<float>()[2] == 2);
  ASSERT(C.data<float>()[10] == 2);
  auto D = A | B;
  ASSERT(D.data<float>()[10] == C.data<float>()[10]);
  C.clear_cache();
  D.clear_cache();
}

TEST(WasmConcat2D) {
  loop_tool::ScopedBackend sb("wasm");
  namespace lz = ::loop_tool::lazy;
  int64_t batch = 2;
  lz::Symbol N("N"), M0("M0"), M1("M1"), M("M");
  lz::Tensor A(N, M0);
  lz::Tensor B(N, M1);
  auto C = A | B;  // different dimensions are concatenated
  A.bind(nullptr, {batch, 5});
  B.bind(nullptr, {batch, 3});
  ASSERT(C.shape()[0] == N);
  ASSERT(C.size(1) == 8);
  for (auto i = 0; i < batch * 5; ++i) {
    A.data<float>()[i] = 11;  // i;
  }
  for (auto i = 0; i < batch * 3; ++i) {
    B.data<float>()[i] = 7;  // i;
  }
  std::cerr << loop_tool::dot(C.ir()) << "\n";
  std::cerr << C.code() << "\n";
  std::cerr << "checking " << C.data<float>()[0] << "\n";
  for (auto i = 0; i < batch * 8; ++i) {
    std::cerr << C.data<float>()[i] << "\n";
  }
  ASSERT(C.data<float>()[6] == 7);
  ASSERT(C.data<float>()[8] == 11);
  C.clear_cache();
}

TEST(WasmPad) {
  loop_tool::ScopedBackend sb("wasm");
  namespace lz = ::loop_tool::lazy;
  lz::Symbol N("N"), Np("Np");
  lz::Tensor X(N);
  // pads both sizes by 1
  lz::Tensor X_pad =
      X.to({Np}, lz::Constraint(Np, N + lz::Expr(1)),
           lz::Constraint(lz::Expr::size(Np), lz::Expr::size(N) + lz::Expr(2)));
  X.bind(nullptr, {5});
  for (auto i = 0; i < 5; ++i) {
    X.data<float>()[i] = i;
  }
  ASSERT(X_pad.size(0) == 7);
  for (auto i = 0; i < 7; ++i) {
    std::cerr << "XPAD " << i << ": " << X_pad.data<float>()[i] << "\n";
  }
  ASSERT(X_pad.data<float>()[2] == 1);
  ASSERT(X_pad.data<float>()[6] == 0);
  X_pad.clear_cache();
}

TEST(WasmPaddedConv) {
  loop_tool::ScopedBackend sb("wasm");
  namespace lz = ::loop_tool::lazy;
  lz::Symbol N("N"), Np("Np"), K("K"), No("No");
  lz::Tensor X(N);
  lz::Tensor W(K);
  auto paddedX =
      X.to({Np}, lz::Constraint(Np, N + lz::Expr(1)),
           lz::Constraint(lz::Expr::size(Np), lz::Expr::size(N) + lz::Expr(2)));

  // implicit constraint -> Np = size(N) + size(K) - 1
  auto expandedX = paddedX.to({No, K}, lz::Constraint(Np, No + K));
  ASSERT(expandedX.shape().size() == 2);
  // ASSERT(expandedX.shape().at(0) == N);
  ASSERT(expandedX.shape().at(1) == K);
  auto Y = (expandedX * W).sum(K);
  X.bind(nullptr, {5});
  W.bind(nullptr, {3});
  for (auto i = 0; i < 5; ++i) {
    X.data<float>()[i] = 1;
  }
  for (auto i = 0; i < 3; ++i) {
    W.data<float>()[i] = 1;
  }
  ASSERT(Y.size(0) == 5);
  Y.data<float>();
  ASSERT(Y.data<float>()[0] == 2);
  ASSERT(Y.data<float>()[2] == 3);
}
