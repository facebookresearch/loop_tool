/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>

#include "test_utils.h"

using namespace loop_tool;
using namespace loop_tool::testing;

TEST(DotDump) {
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

TEST(SetPriority) {
  IR ir;
  auto a = ir.create_var("a");
  auto b = ir.create_var("b");
  auto r0 = ir.create_node("read", {}, {a, b});
  auto r1 = ir.create_node("read", {}, {a, b});
  auto add = ir.create_node("add", {r0, r1}, {a, b});
  auto w = ir.create_node("write", {add}, {a, b});
  ir.set_inputs({r0, r1});
  ir.set_priority(r1, 10);
  LoopTree lt(ir);
  std::cerr << "dumping:\n";
  std::cerr << lt.dump();
}

TEST(NegativeSizes) {
  IR ir;
  auto a = ir.create_var("a");
  auto b = ir.create_var("b");
  auto c = ir.create_var("c");
  auto r0 = ir.create_node("read", {}, {a, b});
  auto r1 = ir.create_node("read", {}, {b, c});
  auto mul = ir.create_node("mul", {r0, r1}, {a, b, c});
  auto add = ir.create_node("add", {mul}, {a, c});
  auto w = ir.create_node("write", {add}, {a, c});
  ir.set_inputs({r0, r1});
  ir.set_priority(r1, 10);
  ir.set_priority(r0, 100);
  ir.set_order(r1, {{b, {-1, 0}}, {c, {-1, 0}}});
  LoopTree lt(ir);
  std::cerr << "dumping:\n";
  std::cerr << lt.dump();
}

TEST(BasicSchedule) {
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
  ir.set_order(r1, {{m, {M, 0}}, {n, {N, 0}}, {k, {K, 0}}});
  ir.set_priority(r1, 10);
  ir.set_priority(r0, 0);
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
  exec(lt, {in0, in1, out});
  float out_ref[M * N];
  ref_mm(in0, in1, M, N, K, out_ref);
  float max_diff = 0;
  for (auto i = 0; i < M * N; ++i) {
    max_diff = std::max(max_diff, std::abs((float)(out_ref[i] - out[i])));
  }
  ASSERT(max_diff < 0.01);
}

TEST(NodeSplit) {
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
