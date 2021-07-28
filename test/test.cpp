/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <chrono>
#include <iostream>
#include <random>

#include "loop_tool/compile.h"
#include "loop_tool/ir.h"

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
    auto lt = LoopTree(ir);
    std::cout << lt.dump();
    lt.walk([&](LoopTree::TreeRef ref, int) {
      if (lt.tree_node(ref).kind != LoopTree::LOOP) {
        return;
      }
      std::cout << "parallel L" << ref << ": ";
      std::cout << trivially_parallel(lt, ref) << "\n";
    });
  }
}
