/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>
#include <loop_tool/dynlib.h>
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
  auto code = C.code();

	std::ofstream("fn_impl.cpp") << code;
	std::system("cc -Wall -Werror -fpic -shared -o fn_impl.so fn_impl.cpp"); // compile
  loop_tool::DynamicLibrary dll("fn_impl.so");
	auto fn = dll.sym<void(*)(void**)>("fn");
	{
		float* A = (float*)calloc(sizeof(float), 16 * 16);
		float* B = (float*)calloc(sizeof(float), 16 * 16);
		float* C = (float*)calloc(sizeof(float), 16 * 16);
		float* C_ref = (float*)calloc(sizeof(float), 16 * 16);
		for (int64_t i = 0; i < 16*16; ++i) {
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
		void *mem[5] = { A, B, C, 0, tmp };
		fn(mem);
		for (int64_t i = 0; i < 16 * 16; ++i) {
      auto diff = std::abs(C[i] - C_ref[i]);
			ASSERT(diff < 0.01) << "difference of " << diff;
		}
	}

  // ensure the symbols are local
	std::ofstream("fn_impl2.cpp") << code;
	std::system("cc -O2 -Wall -Werror -fpic -shared -o fn_impl.so fn_impl2.cpp"); // compile
  loop_tool::DynamicLibrary dll2("fn_impl.so");
	auto fn2 = dll.sym<void(*)(void**)>("fn");
	{
		float* A = (float*)calloc(sizeof(float), 16 * 16);
		float* B = (float*)calloc(sizeof(float), 16 * 16);
		float* C = (float*)calloc(sizeof(float), 16 * 16);
		float* C_ref = (float*)calloc(sizeof(float), 16 * 16);
		for (int64_t i = 0; i < 16*16; ++i) {
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
		void *mem[5] = { A, B, C, 0, tmp };
		fn2(mem);
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
  auto code = C.code();
	std::cerr << code << "\n";
	std::ofstream("fn_impl.cpp") << code;
	std::system("cc -Wall -Werror -fpic -shared -o fn_impl.so fn_impl.cpp"); // compile
  loop_tool::DynamicLibrary dll("fn_impl.so");
	auto fn = dll.sym<void(*)(void**)>("fn");
	{
		float* A = (float*)calloc(sizeof(float), 16 * 16);
		float* B = (float*)calloc(sizeof(float), 16 * 16);
		float* C = (float*)calloc(sizeof(float), 16 * 16);
		float* C_ref = (float*)calloc(sizeof(float), 16 * 16);
		for (int64_t i = 0; i < 16*16; ++i) {
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
		void *mem[5] = { A, B, C, 0, tmp };
		fn(mem);
		for (int64_t i = 0; i < 16 * 16; ++i) {
      auto diff = std::abs(C[i] - C_ref[i]);
			ASSERT(diff < 0.01) << "difference of " << diff;
		}
	}
}
