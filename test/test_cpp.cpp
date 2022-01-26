/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/dynlib.h>
#include <loop_tool/loop_tool.h>

#include <fstream>

#include "test_utils.h"

TEST(CppFromLazy) {
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
  auto compiler = loop_tool::Compiler(C.loop_tree());
  auto code = compiler.gen_string();
  std::string fn_name = "fn_" + std::to_string(compiler.count);

  std::ofstream("/tmp/fn_impl.c") << code;
  std::system(
      "cc -Wall -Werror -fpic -shared -o /tmp/fn_impl.so /tmp/fn_impl.c");  // compile
  loop_tool::DynamicLibrary dll("/tmp/fn_impl.so");
  auto fn = dll.sym<void (*)(void**)>(fn_name.c_str());
  {
    float* A = (float*)calloc(sizeof(float), 16 * 16);
    float* B = (float*)calloc(sizeof(float), 16 * 16);
    float* C = (float*)calloc(sizeof(float), 16 * 16);
    float* C_ref = (float*)calloc(sizeof(float), 16 * 16);
    for (int64_t i = 0; i < 16 * 16; ++i) {
      A[i] = i * 3;
      B[i] = 100 - (i * 2);
    }
    for (int64_t i = 0; i < 16; ++i) {
      for (int64_t j = 0; j < 16; ++j) {
        for (int64_t k = 0; k < 16; ++k) {
          C_ref[i * 16 + j] += A[i * 16 + k] * B[k * 16 + j];
        }
      }
    }
    void* tmp = malloc(sizeof(float) * 16);
    void* mem[5] = {A, B, C, 0, tmp};
    fn(mem);
    for (int64_t i = 0; i < 16 * 16; ++i) {
      auto diff = std::abs(C[i] - C_ref[i]);
      ASSERT(diff < 0.01) << "difference of " << diff;
    }
  }
}

TEST(CppWithTail) {
  namespace lz = ::loop_tool::lazy;
  auto mm = [](lz::Tensor A, lz::Tensor B) {
    auto M = lz::Symbol("m"), N = lz::Symbol("n"), K = lz::Symbol("k");
    auto C = A.as(M, K) * B.as(K, N);
    return C.sum(K);
  };

  lz::Tensor A(16, 16);
  lz::Tensor B(16, 16);
  auto C = mm(A, B);
  auto lt = C.loop_tree();
  std::cerr << '\n';
  std::cerr << lt.dump();
  std::cerr << '\n';
  auto r = lt.children(lt.roots.at(0)).at(0);
  lt = loop_tool::split(lt, r, 10);

  auto a = lt.children(lt.children(lt.roots.at(0)).at(0)).at(0);
  auto b = lt.children(a).at(0);
  lt = loop_tool::swap(lt, a, b);

  std::cerr << '\n';
  std::cerr << lt.dump();

  C.compile();
  C.set(lt);
  auto compiler = loop_tool::Compiler(C.loop_tree());
  auto code = compiler.gen_string();
  std::string fn_name = "fn_" + std::to_string(compiler.count);
  std::cerr << code << "\n";
  std::ofstream("/tmp/fn_impl.c") << code;
  std::system(
      "cc -Wall -Werror -fpic -shared -o /tmp/fn_impl.so /tmp/fn_impl.c");  // compile
  loop_tool::DynamicLibrary dll("/tmp/fn_impl.so");
  auto fn = dll.sym<void (*)(void**)>(fn_name.c_str());
  {
    float* A = (float*)calloc(sizeof(float), 16 * 16);
    float* B = (float*)calloc(sizeof(float), 16 * 16);
    float* C = (float*)calloc(sizeof(float), 16 * 16);
    float* C_ref = (float*)calloc(sizeof(float), 16 * 16);
    for (int64_t i = 0; i < 16 * 16; ++i) {
      A[i] = i * 3;
      B[i] = 100 - (i * 2);
    }
    for (int64_t i = 0; i < 16; ++i) {
      for (int64_t j = 0; j < 16; ++j) {
        for (int64_t k = 0; k < 16; ++k) {
          C_ref[i * 16 + j] += A[i * 16 + k] * B[k * 16 + j];
        }
      }
    }
    void* tmp = malloc(sizeof(float) * 16);
    void* mem[5] = {A, B, C, 0, tmp};
    fn(mem);
    for (int64_t i = 0; i < 16 * 16; ++i) {
      auto diff = std::abs(C[i] - C_ref[i]);
      ASSERT(diff < 0.01) << "difference of " << diff << " at " << i / 16
                          << ", " << i % 16 << " (" << C[i] << " vs expected "
                          << C_ref[i] << ")";
    }
  }
}

TEST(CppView) {
  namespace lz = ::loop_tool::lazy;
  auto padded_conv = [](lz::Tensor X, lz::Tensor W) {
    auto N = lz::Symbol("n"), Np = lz::Symbol("np");
    auto X_pad = X.as(N).pad(N, 1).as(Np);
    auto No = lz::Symbol("no"), K = lz::Symbol("k");
    return (X_pad.to({No, K}, {{Np, No + K}}) * W.as(K)).sum(K);
  };
  lz::Tensor A(16);
  lz::Tensor B(3);
  auto C = padded_conv(A, B);
  auto lt = C.loop_tree();
  std::cerr << '\n';
  std::cerr << lt.dump();
  std::cerr << '\n';
  auto compiler = loop_tool::Compiler(C.loop_tree());
  auto code = compiler.gen_string();
  std::string fn_name = "fn_" + std::to_string(compiler.count);

  std::cerr << code << "\n";
  std::ofstream("/tmp/fn_impl.c") << code;
  std::system(
      "cc -g -O0 -Wall -Werror -fpic -shared -o /tmp/fn_impl.so "
      "/tmp/fn_impl.c");  // compile
  loop_tool::DynamicLibrary dll("/tmp/fn_impl.so");
  auto fn = dll.sym<void (*)(void**)>(fn_name.c_str());
  {
    float* A = (float*)calloc(sizeof(float), 16);
    float* B = (float*)calloc(sizeof(float), 3);
    float* C = (float*)calloc(sizeof(float), 16);
    float* C_ref = (float*)calloc(sizeof(float), 16);
    for (int64_t i = 0; i < 16; ++i) {
      A[i] = i * 3 + 1;
    }
    for (int64_t i = 0; i < 3; ++i) {
      B[i] = 1 - (i * 2);
    }
    for (int64_t i = 0; i < 16; ++i) {
      for (int64_t k = 0; k < 3; ++k) {
        if ((i + k - 1 >= 0) && (i + k - 1 < 16)) {
          C_ref[i] += A[i + k - 1] * B[k];
        }
      }
    }
    void* tmp = malloc(sizeof(float) * 18);
    void* mem[5] = {A, B, C, tmp};
    fn(mem);
    for (int64_t i = 0; i < 16; ++i) {
      auto diff = std::abs(C[i] - C_ref[i]);
      std::cerr << C[i] << " vs " << C_ref[i] << "\n";
      ASSERT(diff < 0.01) << "difference of " << diff;
    }
  }
}
