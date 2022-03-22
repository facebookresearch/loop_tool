/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>

#include "test_utils.h"

namespace lt = loop_tool;
using namespace loop_tool::testing;

TEST(SerializationBasic) {
  namespace lz = ::loop_tool::lazy;
  auto N = lz::Symbol("N");
  auto size = 137;
  lz::Tensor A(size);
  lz::Tensor B(size);
  auto C = A.as(N) + B.as(N);
  rand(A.data<float>(), size);
  rand(B.data<float>(), size);
  const auto& ir = C.ir();
  std::cerr << dot(ir) << "\n";
  auto s_ir = lt::serialize(ir);
  std::cerr << s_ir << "\n";
  const auto& ir_d = lt::deserialize(s_ir);
  C.set(ir_d);
  std::cerr << dot(ir_d) << "\n";
  float max_diff = 0;
  for (auto i = 0; i < size; ++i) {
    auto ref = A.data<float>()[i] + B.data<float>()[i];
    auto diff = std::abs(C.data<float>()[i] - ref);
    max_diff = std::max(max_diff, diff);
  }
  ASSERT(max_diff < 0.01) << "got diff of " << max_diff;
}

TEST(SerializationScheduled) {
  namespace lz = ::loop_tool::lazy;
  auto N = lz::Symbol("N");
  auto size = 138;
  lz::Tensor A(size);
  lz::Tensor B(size);
  auto C = A.as(N) + B.as(N);
  rand(A.data<float>(), size);
  rand(B.data<float>(), size);
  auto ir = C.ir();
  auto v = ir.vars().at(0);
  for (auto n : ir.nodes()) {
    switch (ir.node(n).op()) {
      case lt::Operation::read:
      case lt::Operation::write:
        continue;
      default:
        break;
    }
    ir.set_order(n, {{v, {27, 3}}, {v, {5, 0}}});
    ir.annotate_loop(n, 1, "unroll");
    ir.disable_reuse(n, 1);
  }
  auto dot_before = dot(ir);
  auto s_ir = lt::serialize(ir);
  const auto& ir_d = lt::deserialize(s_ir);
  auto dot_after = dot(ir_d);
  ASSERT(dot_before == dot_after);
  C.set(ir_d);
  float max_diff = 0;
  for (auto i = 0; i < size; ++i) {
    auto ref = A.data<float>()[i] + B.data<float>()[i];
    auto diff = std::abs(C.data<float>()[i] - ref);
    max_diff = std::max(max_diff, diff);
  }
  ASSERT(max_diff < 0.01) << "got diff of " << max_diff;
}

TEST(SerializationDeletedNodes) {
  namespace lz = ::loop_tool::lazy;
  auto N = lz::Symbol("N");
  auto size = 132;
  lz::Tensor A(size);
  lz::Tensor B(size);
  auto C = A.as(N) + B.as(N);
  rand(A.data<float>(), size);
  rand(B.data<float>(), size);
  auto ir = C.ir();
  lt::LoopTree tree(ir);
  {
    auto c = tree.children(tree.roots[0]);
    auto write = c[1];
    tree = copy_input(tree, write, 0);
  }
  {
    auto c = tree.children(tree.roots[0]);
    auto add = c[0];
    tree = copy_input(tree, add, 0);
  }
  {
    auto c = tree.children(tree.roots[0]);
    auto copy = c[c.size() - 2];
    tree = delete_copy(tree, copy);
  }
  {
    auto c = tree.children(tree.roots[0]);
    auto copy = c[0];
    tree = delete_copy(tree, copy);
  }
  C.set(tree);
  ir = C.ir();
  ir.reify_deletions();
  auto dot_before = dot(ir);
  std::cerr << dot_before << "\n";
  auto s_ir = lt::serialize(ir);
  const auto& ir_d = lt::deserialize(s_ir);
  auto dot_after = dot(ir_d);
  ASSERT(dot_before == dot_after);
  C.set(ir_d);
  float max_diff = 0;
  for (auto i = 0; i < size; ++i) {
    auto ref = A.data<float>()[i] + B.data<float>()[i];
    auto diff = std::abs(C.data<float>()[i] - ref);
    max_diff = std::max(max_diff, diff);
  }
  ASSERT(max_diff < 0.01) << "got diff of " << max_diff;
}

TEST(SerializationConv) {
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

  auto dot_before = dot(Y.ir());
  std::cerr << dot_before;
  auto ir = Y.ir();
  auto s = lt::serialize(ir);
  std::cerr << s << "\n";
  auto ir_d = lt::deserialize(s);
  auto dot_after = dot(ir_d);
  std::cerr << dot_after << "\n";
  ASSERT(dot_before == dot_after);
  Y.set(ir_d);
  ASSERT(Y.data<float>()[3] == 3);
}
