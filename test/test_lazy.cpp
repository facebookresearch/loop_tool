/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>
#include <string.h>

#include <chrono>
#include <random>

#include "test_utils.h"

using namespace loop_tool;
using namespace loop_tool::testing;

TEST(LazyBind) {
  namespace lz = ::loop_tool::lazy;
  auto M = lz::Symbol("M");
  auto N = lz::Symbol("N");
  auto K = lz::Symbol("K");
  auto A = lz::Tensor(M, K);
  auto B = lz::Tensor(K, N);
  auto C = A * B;
  ASSERT(C.shape().size() == 3);
  auto D = C.sum(K);
  ASSERT(D.shape().size() == 2);
  ASSERT(D.shape()[0] == M);
  ASSERT(D.shape()[1] == N);
  std::cout << "pass!\n";
  int64_t M_size = 16;
  int64_t N_size = 16;
  int64_t K_size = 16;
  std::vector<float> A_(M_size * K_size);
  std::vector<float> B_(K_size * N_size);
  for (auto m = 0; m < M_size; ++m) {
    for (auto n = 0; n < N_size; ++n) {
      for (auto k = 0; k < K_size; ++k) {
        A_[m * K_size + k] = 1 + m * k * 1.8 / 100;
        B_[k * N_size + n] = 1 + n * k * 1.8 / 100;
      }
    }
  }
  A.bind(A_.data(), {M_size, K_size});
  B.bind(B_.data(), {K_size, N_size});

  auto d = D.data<float>();
  for (auto m = 0; m < M_size; ++m) {
    for (auto n = 0; n < N_size; ++n) {
      std::cout << d[m * N_size + n] << " ";
    }
    std::cout << "\n";
  }
}
TEST(LazyData) {
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

  lz::Tensor C_ref(M, N);
  ref_mm(A.data<float>(), B.data<float>(), M, N, K, C_ref.data<float>());

  float max_diff = 0;
  for (auto i = 0; i < M * N; ++i) {
    max_diff = std::max(max_diff,
                        std::abs(C.data<float>()[i] - C_ref.data<float>()[i]));
  }
  std::cout << "max diff " << max_diff << "\n";
}

TEST(LazyCaching) {
  std::cout << "doing mm\n";
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

  for (auto i = 0; i < 2; ++i) {
    lz::Tensor A(M, K);
    lz::Tensor B(K, N);
    rand(A.data<float>(), M * K);
    rand(B.data<float>(), K * N);
    auto C = mm(A, B);
    (void)C.data<float>();

    lz::Tensor C_ref(M, N);
    ref_mm(A.data<float>(), B.data<float>(), M, N, K, C_ref.data<float>());

    float max_diff = 0;
    for (auto i = 0; i < M * N; ++i) {
      max_diff = std::max(
          max_diff, std::abs(C.data<float>()[i] - C_ref.data<float>()[i]));
    }
    std::cout << "max diff " << max_diff << "\n";
  }

  std::cout << "many mm\n";
  auto iters = 100;
  auto start = std::chrono::steady_clock::now();
  for (auto i = 0; i < iters; ++i) {
    lz::Tensor A(M, K);
    lz::Tensor B(K, N);
    // rand(A.data<float>(), M * K);
    // rand(B.data<float>(), K * N);
    auto C = mm(A, B);
    (void)C.data<float>();
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cerr << iters / diff.count() << " iters/sec\n";
}
TEST(LazyAdd) {
  std::cout << "doing add\n";
  namespace lz = ::loop_tool::lazy;
  auto add = [](lz::Tensor A, lz::Tensor B) {
    auto N = lz::Symbol("N");
    auto C = A.as(N) + B.as(N);
    return C;
  };

  int64_t size = 4;

  {
    lz::Tensor A(size);
    lz::Tensor B(size);
    rand(A.data<float>(), size);
    rand(B.data<float>(), size);
    for (auto i = 0; i < size; ++i) {
      std::cout << A.data<float>()[i] + B.data<float>()[i] << "\n";
    }
    float max_diff = 0;
    auto C = add(A, B);
    for (auto i = 0; i < size; ++i) {
      std::cout << A.data<float>()[i] + B.data<float>()[i] << "\n";
    }
    for (auto i = 0; i < size; ++i) {
      auto ref = A.data<float>()[i] + B.data<float>()[i];
      auto diff = std::abs(C.data<float>()[i] - ref);
      max_diff = std::max(max_diff, diff);
      std::cout << "ref: " << ref << " vs " << C.data<float>()[i] << "\n";
      // std::cout << ref << ":" << C.data<float>()[i] << "(" << diff << ") ";
    }
    std::cout << "\n";
    std::cout << "max diff " << max_diff << "\n";
    ASSERT(max_diff < 0.01) << "got diff of " << max_diff;
  }
  return;

  std::vector<float> A_(size);
  std::vector<float> B_(size);
  rand(A_.data(), size);
  rand(B_.data(), size);
  auto iters = 10000;
  auto start = std::chrono::steady_clock::now();
  for (auto i = 0; i < iters; ++i) {
    lz::Tensor A(A_.data(), {size});
    lz::Tensor B(B_.data(), {size});
    auto C = add(A, B);
    (void)C.data<float>();
    auto diff = std::abs(C.data<float>()[0] - (A_[0] + B_[0]));
    ASSERT(diff < 0.01) << "got: " << C.data<float>()[0]
                        << " expected: " << (A_[0] + B_[0]);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cerr << iters / diff.count() << " iters/sec\n";
}

TEST(LazyLoopTree) {
  namespace lz = ::loop_tool::lazy;
  auto add = [](lz::Tensor A, lz::Tensor B) {
    auto N = lz::Symbol("N");
    auto C = A.as(N) + B.as(N);
    return C;
  };

  int64_t size = 4;

  lz::Tensor A(size);
  lz::Tensor B(size);
  rand(A.data<float>(), size);
  rand(B.data<float>(), size);
  float max_diff = 0;
  auto C = add(A, B);
  for (auto i = 0; i < size; ++i) {
    auto ref = A.data<float>()[i] + B.data<float>()[i];
    auto diff = std::abs(C.data<float>()[i] - ref);
    max_diff = std::max(max_diff, diff);
  }
  std::cout << "max diff " << max_diff << "\n";

  auto lt = C.loop_tree();
  ASSERT(lt.roots.size() > 0);
  std::cout << "schedule is:\n" << A.loop_tree().dump();
}

TEST(LazyStackMemory) {
  namespace lz = loop_tool::lazy;
  lz::Tensor A(128);
  lz::Tensor B(128);

  memset(A.data<float>(), 0, sizeof(float) * 128);
  for (auto i = 0; i < 128; ++i) {
    B.data<float>()[i] = i;
  }

  lz::Symbol N;
  auto C = A.as(N) + B.as(N);

  std::cout << C.data<float>()[14] << "\n";  // prints 14

  float scale = 2.f;
  auto S = lz::Tensor(&scale, {1});  // user data is fine, it isn't copied
  auto D = B * S;  // broadcasts for us, no need to use lz::Tensor::as()

  ASSERT(D.data<float>()[7] == 14);
  std::cout << D.loop_tree().dump() << "\n";
}

TEST(LazyFill) {
  namespace lz = ::loop_tool::lazy;
  lz::Tensor A(1);
  A.data<float>()[0] = 4.5;
  lz::Symbol M, N;
  lz::Tensor B(M, N);
  B.bind(nullptr, {8, 8});
  for (auto i = 0; i < 64; ++i) {
    B.data<float>()[i] = 2;
  }
  auto C = A * B;
  ASSERT(std::abs(C.data<float>()[7] - 9) < 0.001);
}

TEST(LazyTranspose) {
  namespace lz = ::loop_tool::lazy;
  lz::Symbol M, N;
  lz::Tensor A(M, N);
  auto B = A.to({N});  // slice and transpose
  // B.shape() == {N}
  B = A.to({N, M});
  // create_node("view", {A}, {N, M});
  // create_node("view", {A}, {K}, {K = N * size(M) + M});
  // B.shape() == {N, M}
  lz::Symbol K;
  // K = N * size(M) + M
  // M = K - N * size(M)
  B = A.to({K}, lz::Constraint(K, N * lz::Expr::size(M) + M));  // ugly flatten
  A.bind(nullptr, {8, 8});
  ASSERT(B.size(0) == 64);
}

TEST(LazyConv) {
  namespace lz = ::loop_tool::lazy;
  lz::Symbol N("N"), N_o("N_o"), K("K");
  lz::Tensor A(N);
  lz::Tensor W(K);

  lz::Tensor X = A.to({N_o, K}, lz::Constraint(N, N_o + K));
  auto Y = (X * W).sum(K);
  Y.bind(nullptr, {8});  // we can infer the size of A from this
  W.bind(nullptr, {3});
  float A_data[10] = {0};
  A.bind(A_data, {10});
  for (auto i = 0; i < 10; ++i) {
    A.data<float>()[i] = 1;
  }
  for (auto i = 0; i < 3; ++i) {
    W.data<float>()[i] = 1;
  }
  ASSERT(X.shape().size() == 2);
  ASSERT(X.shape()[0] == N_o);
  ASSERT(X.shape()[1] == K);
  ASSERT(A.size(0) == 10);
  std::cerr << "output data: \n";
  for (auto i = 0; i < 8; ++i) {
    std::cerr << Y.data<float>()[i] << " ";
  }
  std::cerr << "\n";
  ASSERT(Y.data<float>()[3] == 3) << "found " << Y.data<float>()[3];
}

TEST(LazyConvStride) {
  namespace lz = ::loop_tool::lazy;
  lz::Symbol N("N"), N_o("N_o"), K("K");
  lz::Tensor A(N);
  lz::Tensor W(K);
  lz::Tensor X = A.to({N_o, K}, lz::Constraint(N, lz::Expr(2) * N_o + K));
  auto Y = (X * W).sum(K);
  Y.bind(nullptr, {8});  // we can infer the size of A from this
  W.bind(nullptr, {3});
  float A_data[17] = {0};
  A.bind(A_data, {17});
  for (auto i = 0; i < 10; ++i) {
    A.data<float>()[i] = 1;
  }
  for (auto i = 0; i < 3; ++i) {
    W.data<float>()[i] = 1;
  }
  ASSERT(X.shape().size() == 2);
  ASSERT(X.shape()[0] == N_o);
  ASSERT(X.shape()[1] == K);
  ASSERT(A.size(0) == 17) << "found size to be " << A.size(0);
  ASSERT(Y.data<float>()[3] == 3);
}

TEST(LazyConcat1D) {
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
}

TEST(LazyConcat2D) {
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
}

TEST(LazyPadOnly) {
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
}

TEST(LazyPaddedConv) {
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
  ASSERT(Y.size(0) == 5);
  auto ir = Y.ir();
  for (auto n : ir.nodes()) {
    switch (ir.node(n).op()) {
      case Operation::read:
      case Operation::write:
      case Operation::view:
        ir.set_order(n, {});
        if (n == 1) {
          auto vi = ir.node(ir.node(n).inputs().at(0)).vars().at(0);
          auto v0 = ir.node(n).vars().at(0);
          // auto v1 = ir.node(n).vars().at(1);
          // ir.set_order(n, {
          //  {v0, {5, 0}},
          //  {v1, {3, 0}}
          //});
          // ir.set_order(n, {
          //  {v0, {7, 0}}
          //});
          // ir.set_order(n, {
          //  {vi, {5, 0}}
          //});
        }
        break;
      default:
        break;
    }
  }
  LoopTree loop_tree(ir);
  auto f = Compiler(loop_tree).gen_exec(-1);
  // ASSERT(0);

  Y.compile();
  Y.set(ir);
  // LoopTree lt(ir); std::cout << lt.dump() << "\n";
  Y.data<float>();
}

TEST(LazySplit) {
  namespace lz = ::loop_tool::lazy;
  lz::Symbol M("M"), K("K"), N("N");
  lz::Tensor X(128);
  X = X.as(N);
  X.to({M, K}, lz::Constraint(N, M * lz::Expr::size(K) + K),
       lz::Constraint(lz::Expr::size(K), lz::Expr(2)));
  (void)X.data<float>();
}
