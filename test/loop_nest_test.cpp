/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>

#include "test_utils.h"

TEST(LoopToolBackend) {
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
  std::cerr << d[0] << "\n";
}
