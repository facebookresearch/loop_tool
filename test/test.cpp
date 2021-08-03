/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <chrono>
#include <iostream>
#include <random>

#include "loop_tool/backend.h"
#include "loop_tool/compile.h"
#include "loop_tool/ir.h"
#include "loop_tool/lazy.h"

using namespace loop_tool;

void rand(float *data, int N) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::normal_distribution<> dist(2, 2);
  for (auto i = 0; i < N; ++i) {
    data[i] = dist(e2);
  }
}

// assumes LCA=K, LCB=N
void ref_mm(const float *A, const float *B, int M, int N, int K, float *C,
            float alpha = 0) {
  for (auto n = 0; n < N; ++n) {
    for (auto m = 0; m < M; ++m) {
      float tmp = 0;
      for (auto k = 0; k < K; ++k) {
        tmp += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = alpha * C[m * N + n] + tmp;
    }
  }
}

int main() {
  // std::cerr << "1\n";
  // std::cerr << "2\n";
  //{
  //  IR ir;
  //  auto a = ir.create_var("a");
  //  auto b = ir.create_var("b");
  //  auto r0 = ir.create_node("read", {}, {a, b});
  //  auto r1 = ir.create_node("read", {}, {a, b});
  //  auto add = ir.create_node("add", {r0, r1}, {a, b});
  //  auto w = ir.create_node("write", {add}, {a, b});
  //  ir.set_inputs({r0, r1});
  //  ir.set_priority(r1, 10);
  //  LoopTree lt(ir);
  //  std::cerr << "dumping:\n";
  //  std::cerr << lt.dump();
  //}
  //{
  //  IR ir;
  //  auto a = ir.create_var("a");
  //  auto b = ir.create_var("b");
  //  auto c = ir.create_var("c");
  //  auto r0 = ir.create_node("read", {}, {a, b});
  //  auto r1 = ir.create_node("read", {}, {b, c});
  //  auto mul = ir.create_node("mul", {r0, r1}, {a, b, c});
  //  auto add = ir.create_node("add", {mul}, {a, c});
  //  auto w = ir.create_node("write", {add}, {a, c});
  //  ir.set_inputs({r0, r1});
  //  ir.set_priority(r1, 10);
  //  ir.set_priority(r0, 100);
  //  ir.set_order(r1, {{b, -1}, {c, -1}});
  //  LoopTree lt(ir);
  //  std::cerr << "dumping:\n";
  //  std::cerr << lt.dump();
  //}

  //{
  //  IR ir;
  //  auto a = ir.create_var("a");
  //  auto b = ir.create_var("b");
  //  auto c = ir.create_var("c");
  //  auto r0 = ir.create_node("read", {}, {a, b});
  //  auto r1 = ir.create_node("read", {}, {b, c});
  //  auto mul = ir.create_node("mul", {r0, r1}, {a, b, c});
  //  auto add = ir.create_node("add", {mul}, {a, c});
  //  auto w = ir.create_node("write", {add}, {a, c});
  //  ir.set_inputs({r0, r1});
  //  ir.set_priority(r1, 10);
  //  ir.set_priority(r0, 100);
  //  ir.set_order(r0, {{b, 20}, {b, 35}, {b, 7}, {a, 32}});
  //  ir.set_order(r1, {{b, 20}, {b, 35}, {c, 30}});
  //  LoopTree lt(ir);
  //  std::cerr << "dumping:\n";
  //  std::cerr << lt.dump();
  //}

  //{
  //  IR ir;
  //  auto a = ir.create_var("a");
  //  auto b = ir.create_var("b");
  //  auto r0 = ir.create_node("read", {}, {a, b});
  //  auto r1 = ir.create_node("read", {}, {a, b});
  //  auto add = ir.create_node("add", {r1, r0}, {a, b});
  //  auto w = ir.create_node("write", {add}, {a, b});
  //  for (auto v : { r0, r1, add, w }) {
  //    ir.set_order(v, {{a, {3, 1}}, {b, {5, 0}}});
  //  }
  //  ir.set_inputs({r0, r1});
  //  ir.set_outputs({w});
  //  LoopTree lt(ir);
  //  float in0[16];
  //  float in1[16];
  //  float out[16];
  //  for (auto i = 0; i < 16; ++i) {
  //    in0[i] = i;
  //    in1[i] = 3;
  //  }
  //  exec(lt, {in0, in1, out});
  //  for (auto i = 0; i < 16; ++i) {
  //    std::cerr << out[i] << "\n";
  //  }
  //}
  if (0) {
    IR ir;
    auto a = ir.create_var("a");
    auto r0 = ir.create_node("read", {}, {a});
    auto r1 = ir.create_node("read", {}, {a});
    auto add = ir.create_node("add", {r1, r0}, {a});
    auto w = ir.create_node("write", {add}, {a});
    for (auto v : {r0, r1, add, w}) {
      ir.set_order(v, {{a, {3, 1}}, {a, {2, 1}}, {a, {2, 0}}});
    }
    ir.set_order(add, {{a, {3, 1}}, {a, {5, 0}}});
    ir.set_inputs({r0, r1});
    ir.set_outputs({w});
    LoopTree lt(ir);
    std::cerr << lt.dump();
    float in0[16];
    float in1[16];
    float out[16];
    for (auto i = 0; i < 16; ++i) {
      in0[i] = i;
      in1[i] = 3;
    }
    exec(lt, {in0, in1, out});
    for (auto i = 0; i < 16; ++i) {
      std::cerr << out[i] << "\n";
    }
  }
  if (0) {
    IR ir;
    constexpr int N = 16;
    auto a = ir.create_var("a");
    auto r0 = ir.create_node("read", {}, {a});
    auto r1 = ir.create_node("read", {}, {a});
    auto add = ir.create_node("add", {r1, r0}, {a});
    auto w = ir.create_node("write", {add}, {a});
    for (auto v : {r0, r1, add, w}) {
      ir.set_order(v, {{a, {N, 0}}});
    }
    ir.set_inputs({r0, r1});
    ir.set_outputs({w});
    LoopTree lt(ir);
    std::cerr << lt.dump();
    float in0[N];
    float in1[N];
    float out[N];
    for (auto i = 0; i < N; ++i) {
      in0[i] = i;
      in1[i] = 3;
    }
    exec(lt, {in0, in1, out});
    for (auto i = 0; i < 16; ++i) {
      std::cerr << out[i] << "\n";
    }
  }
  if (0) {
    IR ir;
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;
    auto m = ir.create_var("m");
    auto n = ir.create_var("n");
    auto k = ir.create_var("k");

    auto r0 = ir.create_node("read", {}, {m, k});
    auto r1 = ir.create_node("read", {}, {k, n});

    auto mul = ir.create_node("mul", {r1, r0}, {m, k, n});
    auto add = ir.create_node("add", {mul}, {m, n});

    auto w = ir.create_node("write", {add}, {m, n});

    ir.set_order(r0, {{m, {M, 0}}, {k, {K, 0}}});
    // ir.set_order(r1, {{k, {K, 0}}, {n, {N, 0}}});
    // ir.set_order(r0, {{k, {K, 0}}, {m, {M, 0}}});
    ir.set_order(r1, {{m, {M, 0}}, {n, {N, 0}}, {k, {K, 0}}});
    ir.set_priority(r1, 10);
    ir.set_priority(r0, 0);
    // ir.set_order(mul, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
    // ir.set_order(add, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
    ir.set_order(mul, {{m, {M, 0}}, {n, {N, 0}}, {k, {K, 0}}});
    ir.set_order(add, {{m, {M, 0}}, {n, {N, 0}}, {k, {K, 0}}});
    ir.set_order(w, {{m, {M, 0}}, {n, {N, 0}}});
    ir.set_inputs({r0, r1});
    ir.set_outputs({w});
    LoopTree lt(ir);
    std::cerr << lt.dump();
    float in0[M * K];
    float in1[N * K];
    float out[M * N];
    rand(in0, M * K);
    rand(in1, N * K);
    for (auto i = 0; i < M * N; ++i) {
      // in0[i] = 1;
      // in1[i] = i;
    }
    exec(lt, {in0, in1, out});
    // for (auto i = 0; i < 4; ++i) {
    //  std::cerr << out[i] << "\n";
    //}
    float out_ref[M * N];
    ref_mm(in0, in1, M, N, K, out_ref);
    float max_diff = 0;
    for (auto i = 0; i < M * N; ++i) {
      max_diff = std::max(max_diff, std::abs(out_ref[i] - out[i]));
      // std::cerr << out[i] << " vs ref " << out_ref[i] << "\n";
    }
    std::cerr << "max diff " << max_diff << "\n";
  }
  if (0) {
    IR ir;
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;
    auto m = ir.create_var("m");
    auto n = ir.create_var("n");
    auto k = ir.create_var("k");

    auto r0 = ir.create_node("read", {}, {m, k});
    auto r1 = ir.create_node("read", {}, {k, n});

    auto mul = ir.create_node("mul", {r1, r0}, {m, k, n});
    auto add = ir.create_node("add", {mul}, {m, n});

    auto w = ir.create_node("write", {add}, {m, n});

    ir.set_order(r0, {{m, {M, 0}}, {k, {K, 0}}});
    // ir.set_order(r1, {{k, {K, 0}}, {n, {N, 0}}});
    // ir.set_order(r0, {{k, {K, 0}}, {m, {M, 0}}});
    ir.set_order(r1, {{m, {M, 0}}, {n, {N, 0}}, {k, {K, 0}}});
    ir.set_priority(r1, 10);
    ir.set_priority(r0, 0);
    // ir.set_order(mul, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
    // ir.set_order(add, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
    ir.set_order(mul, {{m, {M, 0}}, {n, {N, 0}}, {k, {K, 0}}});
    ir.set_order(add, {{m, {3, 1}}, {n, {N, 0}}, {k, {K, 0}}, {m, {5, 0}}});
    // ir.set_order(add, {{m, {2, 0}}, {n, {N, 0}}, {k, {K, 0}}, {m, {2, 0}}});
    ir.set_order(w, {{m, {M, 0}}, {n, {N, 0}}});
    ir.set_inputs({r0, r1});
    ir.set_outputs({w});
    LoopTree lt(ir);
    std::cerr << lt.dump();
    float in0[M * K];
    float in1[N * K];
    float out[M * N];
    rand(in0, M * K);
    rand(in1, N * K);
    for (auto i = 0; i < M * N; ++i) {
      // in0[i] = 1;
      // in1[i] = i;
    }
    exec(lt, {in0, in1, out});
    for (auto i = 0; i < 4; ++i) {
      // std::cerr << out[i] << "\n";
    }
    float out_ref[M * N];
    ref_mm(in0, in1, M, N, K, out_ref);
    float max_diff = 0;
    for (auto i = 0; i < M * N; ++i) {
      max_diff = std::max(max_diff, std::abs(out_ref[i] - out[i]));
      // std::cerr << out[i] << " vs ref " << out_ref[i] << "\n";
    }
    std::cerr << "max diff " << max_diff << "\n";
  }
  if (0) {
    IR ir;
    constexpr int N = 128;
    auto a = ir.create_var("a");
    auto r0 = ir.create_node("read", {}, {a});
    auto r1 = ir.create_node("read", {}, {a});
    auto add = ir.create_node("add", {r0, r1}, {a});
    auto w = ir.create_node("write", {add}, {a});
    for (auto v : {r0, r1, add, w}) {
      ir.set_order(v, {{a, {N, 0}}});
    }
    ir.set_inputs({r0, r1});
    ir.set_outputs({w});
    ir.set_priority(r1, 10);
    LoopTree lt(ir);
    auto start = std::chrono::steady_clock::now();
    auto iters = 1000;
    for (auto i = 0; i < iters; ++i) {
      lt = LoopTree(ir);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cerr << iters / diff.count() << " iters/sec\n";
    std::cerr << "dumping:\n";
    std::cerr << lt.dump();
    float in0[N];
    float in1[N];
    float out[N];
    rand(in0, N);
    rand(in1, N);
    exec(lt, {in0, in1, out});
    float out_ref[N];
    for (auto i = 0; i < N; ++i) {
      out_ref[i] = in0[i] + in1[i];
    }
    float max_diff = 0;
    for (auto i = 0; i < N; ++i) {
      max_diff = std::max(max_diff, std::abs(out_ref[i] - out[i]));
    }
    std::cerr << "max diff " << max_diff << "\n";
  }
  if (0) {
    IR ir;
    constexpr int N = 128;
    auto a = ir.create_var("a");
    auto r0 = ir.create_node("read", {}, {a});
    auto r1 = ir.create_node("read", {}, {a});
    auto add = ir.create_node("add", {r0, r1}, {a});
    auto w = ir.create_node("write", {add}, {a});
    for (auto v : {r0, r1, add, w}) {
      ir.set_order(v, {{a, {N, 0}}});
      ir.disable_reuse(v, 0);
    }
    ir.set_inputs({r0, r1});
    ir.set_outputs({w});
    ir.set_priority(r1, 10);
    LoopTree lt(ir);
    auto start = std::chrono::steady_clock::now();
    auto iters = 1000;
    for (auto i = 0; i < iters; ++i) {
      lt = LoopTree(ir);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cerr << iters / diff.count() << " iters/sec\n";
    std::cerr << "dumping:\n";
    std::cerr << lt.dump();
    float in0[N];
    float in1[N];
    float out[N];
    rand(in0, N);
    rand(in1, N);
    exec(lt, {in0, in1, out});
    float out_ref[N];
    for (auto i = 0; i < N; ++i) {
      out_ref[i] = in0[i] + in1[i];
    }
    float max_diff = 0;
    for (auto i = 0; i < 6; ++i) {
      max_diff = std::max(max_diff, std::abs(out_ref[i] - out[i]));
      std::cerr << "inp " << in0[i] << " + " << in1[i] << " ";
      std::cerr << "out " << out[i] << " vs " << out_ref[i] << "\n";
    }
    std::cerr << "max diff " << max_diff << "\n";
  }
  if (0) {
    IR ir;
    constexpr int N = 16;
    auto a = ir.create_var("a");
    auto b = ir.create_var("b");
    auto r = ir.create_node("read", {}, {a, b});
    auto add = ir.create_node("add", {r}, {});
    auto w = ir.create_node("write", {add}, {});
    ir.set_inputs({r});
    ir.set_outputs({w});
    std::cerr << LoopTree(ir).dump() << "\n";
    std::cerr << dot(ir) << "\n";
    ir = split_node(ir, add, {b});
    std::cerr << " -- split -- \n";
    std::cerr << LoopTree(ir).dump() << "\n";
    std::cerr << dot(ir) << "\n";
  }
  {
    IR ir;
    constexpr int N = 16;
    auto a = ir.create_var("a");
    auto b = ir.create_var("b");
    auto r = ir.create_node("read", {}, {a, b});
    auto add = ir.create_node("add", {r}, {});
    auto w = ir.create_node("write", {add}, {});
    ir.set_inputs({r});
    ir.set_outputs({w});
    ir = split_node(ir, add, {b});

    for (auto n : ir.nodes()) {
      std::vector<std::pair<IR::VarRef, IR::LoopSize>> sched;
      for (auto v : ir.all_vars(n)) {
        sched.emplace_back(std::pair<IR::VarRef, IR::LoopSize>{v, {N, 0}});
      }
      ir.set_order(n, sched);
    }

    auto lt = LoopTree(ir);
    lt.walk([&](LoopTree::TreeRef ref, int) {
      if (trivially_parallel(lt, ref)) {
        lt.annotate(ref, "cpu_parallel");
      }
    });
    std::cout << lt.dump() << "\n";

    auto cc = getBackends().at("cpu")->compile(lt, {}, -1);
    std::vector<float> input(N * N);
    for (auto i = 0; i < N * N; ++i) {
      input[i] = i;
    }
    std::vector<float> output(1);
    cc->run({input.data(), output.data()}, true);
    std::cout << "sum of vals from 0 to " << (N * N - 1) << " is " << output[0]
              << "\n";
  }
  {
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
    size_t M_size = 16;
    size_t N_size = 16;
    size_t K_size = 16;
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
  {
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
      max_diff = std::max(
          max_diff, std::abs(C.data<float>()[i] - C_ref.data<float>()[i]));
    }
    std::cout << "max diff " << max_diff << "\n";
  }

  {
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
  {
    std::cout << "doing add\n";
    namespace lz = ::loop_tool::lazy;
    auto add = [](lz::Tensor A, lz::Tensor B) {
      auto N = lz::Symbol("N");
      auto C = A.as(N) + B.as(N);
      return C;
    };

    auto size = 4;

    {
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
    }

    std::vector<float> A_(size);
    std::vector<float> B_(size);
    rand(A_.data(), size);
    rand(B_.data(), size);
    auto iters = 1000;
    auto start = std::chrono::steady_clock::now();
    for (auto i = 0; i < iters; ++i) {
      lz::Tensor A(A_.data(), {size});
      lz::Tensor B(B_.data(), {size});
      auto C = add(A, B);
      (void)C.data<float>();
      ASSERT(C.data<float>()[0] == A_[0] + B_[0]);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cerr << iters / diff.count() << " iters/sec\n";
  }
}
