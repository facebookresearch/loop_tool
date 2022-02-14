/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>

#include <fstream>

#include "backends/wasm/wasm.h"
#include "test_utils.h"

TEST(WasmBasic) {
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
  auto C = mm(A, B);
  auto wc = loop_tool::WebAssemblyCompiler(C.loop_tree());
  auto bytes = wc.emit();
  std::ofstream wasm("out.wasm", std::ios::binary);
  wasm.write((char*)bytes.data(), bytes.size());
}
