/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/wasm.h"

#include <loop_tool/loop_tool.h>

#include <fstream>

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

TEST(WasmUnroll) {
  namespace lz = ::loop_tool::lazy;
  auto N = 4;
  auto n = lz::Symbol("N");

  lz::Tensor A(N);
  lz::Tensor B(N);
  auto C = A.as(n) + B.as(n);
  auto lt = C.loop_tree();

  lt = split(lt, lt.roots.at(0), 2);
  lt =
      disable_reuse(lt, lt.children(lt.roots.at(0)).at(0), lt.ir.nodes().at(2));
  // lt.annotate(lt.roots.at(0), "unroll");
  std::cerr << "\n" << lt.dump();
  auto wc = loop_tool::WebAssemblyCompiler(lt);
  auto bytes = wc.emit();
  std::ofstream wasm("out.wasm", std::ios::binary);
  wasm.write((char*)bytes.data(), bytes.size());
}
