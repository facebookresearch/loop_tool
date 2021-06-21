/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "cuda_backend.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvrtc.h>
#include <random>

float *cuda_rand(int N) {
  void *ptr = nullptr;
  auto s = N * sizeof(float);
  auto err = cudaMallocManaged(&ptr, s); // N * sizeof(float));
  gpuErrchk(err);
  float *data = (float *)ptr;
  std::random_device rd;
  std::mt19937 e2(rd());
  std::normal_distribution<> dist(2, 2);
  for (auto i = 0; i < N; ++i) {
    data[i] = dist(e2);
  }
  return data;
}

void cuda_exec(const LoopTree &lt, const std::vector<void *> &memory,
               const std::unordered_set<LoopTree::TreeRef> &threaded = {-1}) {
  CompiledCuda cc(lt, threaded);
  cc(memory);
}

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

void run(int T, int V) {
  IR ir;
  constexpr int N = 1024 * 1024;
  if (T * V > N) {
    return;
  }
  std::cerr << "Threading size " << T << " vec " << V << "\n";
  auto a = ir.create_var("a");
  auto r0 = ir.create_node("read", {}, {a});
  auto r1 = ir.create_node("read", {}, {a});
  auto add = ir.create_node("add", {r0, r1}, {a});
  auto w = ir.create_node("write", {add}, {a});
  for (auto v : {r0, r1, add, w}) {
    ir.set_order(v, {{a, {N / V / T, 0}}, {a, {T, 0}}, {a, {V, 0}}});
    ir.disable_reuse(v, 2);
  }
  ir.set_inputs({r0, r1});
  ir.set_outputs({w});
  LoopTree lt(ir);
  void *ptr = nullptr;
  auto err = cudaMallocManaged(&ptr, N);
  gpuErrchk(err);
  auto in0 = cuda_rand(N);
  auto in1 = cuda_rand(N);
  auto out = cuda_rand(N);
  assert(lt.roots.size() == 1);
  std::unordered_set<LoopTree::TreeRef> threaded;
  for (auto c : lt.node(lt.roots.at(0)).children) {
    threaded.insert(c);
    // for (auto c2 : lt.node(c).children) {
    //  ca.unrolled.insert(c2);
    //}
    // ca.unrolled.insert(c);
  }
  // ca.unrolled.insert(lt.node(lt.roots.at(0)).children.at(2));
  // ca.threaded.insert(lt.node(lt.roots.at(0)).children.at(0));
  // ca.threaded[lt.roots.at(0)] = -1;

  cuda_exec(lt, {in0, in1, out}, threaded);
  auto max_diff = 0.f;
  for (auto i = 0; i < N; ++i) {
    max_diff = std::max(std::abs(in0[i] + in1[i] - out[i]), max_diff);
  }
  assert(max_diff < 0.01);
  // std::cerr << "max diff " << max_diff << "\n";
  cudaFree(in0);
  cudaFree(in1);
  cudaFree(out);
}

void test_mm(int M, int N, int K) {
  IR ir;
  auto m = ir.create_var("m");
  auto n = ir.create_var("n");
  auto k = ir.create_var("k");

  auto r0 = ir.create_node("read", {}, {m, k});
  auto r1 = ir.create_node("read", {}, {k, n});

  auto mul = ir.create_node("mul", {r1, r0}, {m, k, n});
  auto add = ir.create_node("add", {mul}, {m, n});

  auto w = ir.create_node("write", {add}, {m, n});
  int N_unroll = 4;
  int K_unroll = 4;
  ir.set_order(r0, {{m, {M, 0}}, {k, {K / K_unroll, 0}}, {k, {K_unroll, 0}}});
  // ir.set_order(r1, {{k, {K, 0}}, {n, {N, 0}}});
  // ir.set_order(r0, {{k, {K, 0}}, {m, {M, 0}}});
  ir.set_order(r1, {{m, {M, 0}},
                    {n, {N / N_unroll, 0}},
                    {k, {K / K_unroll, 0}},
                    {k, {K_unroll, 0}},
                    {n, {N_unroll, 0}}});
  ir.set_priority(r1, 10);
  ir.set_priority(r0, 0);
  // ir.set_order(mul, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
  // ir.set_order(add, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
  ir.set_order(mul, {{m, {M, 0}},
                     {n, {N / N_unroll, 0}},
                     {k, {K / K_unroll, 0}},
                     {k, {K_unroll, 0}},
                     {n, {N_unroll, 0}}});
  ir.set_order(add, {{m, {M, 0}},
                     {n, {N / N_unroll, 0}},
                     {k, {K / K_unroll, 0}},
                     {k, {K_unroll, 0}},
                     {n, {N_unroll, 0}}});
  ir.set_order(w, {{m, {M, 0}}, {n, {N / N_unroll, 0}}, {n, {N_unroll, 0}}});
  ir.set_inputs({r0, r1});
  ir.set_outputs({w});
  LoopTree lt(ir);
  auto in0 = cuda_rand(M * K);
  auto in1 = cuda_rand(N * K);
  auto out = cuda_rand(M * N);
  std::cerr << lt.dump();
  std::unordered_set<LoopTree::TreeRef> threaded;
  threaded.insert(lt.roots.at(0));
  threaded.insert(lt.node(lt.roots.at(0)).children.at(0));
  auto sinnermost =
      lt.node(lt.node(lt.node(lt.roots.at(0)).children.at(0)).children.at(0))
          .children.at(0);
  auto innermost =
      lt.node(lt.node(lt.node(lt.node(lt.roots.at(0)).children.at(0))
                          .children.at(0))
                  .children.at(0))
          .children.at(0);
  // ca.unrolled.insert(innermost);
  // ca.unrolled.insert(sinnermost);
  // ca.unrolled.insert(lt.node(lt.roots.at(0)).children.at(0))
  for (auto i = 0; i < M * N; ++i) {
    out[i] = 0;
  }
  cuda_exec(lt, {in0, in1, out}, threaded);
  auto out_ref = cuda_rand(M * N);
  ref_mm(in0, in1, M, N, K, out_ref);
  float max_diff = 0;
  for (auto i = 0; i < M * N; ++i) {
    max_diff = std::max(max_diff, std::abs(out_ref[i] - out[i]));
    // std::cerr << "out: " << out[i]<< " vs " << out_ref[i] << "\n";
  }
  std::cerr << "max diff " << max_diff << "\n";
}

void test_mm2(int M, int N, int K) {
  IR ir;
  auto m = ir.create_var("m");
  auto n = ir.create_var("n");
  auto k = ir.create_var("k");

  auto r0 = ir.create_node("read", {}, {m, k});
  auto r1 = ir.create_node("read", {}, {k, n});

  auto mul = ir.create_node("mul", {r1, r0}, {m, k, n});
  auto add = ir.create_node("add", {mul}, {m, n});

  auto w = ir.create_node("write", {add}, {m, n});
  int N_unroll = 4;
  int M_unroll = 4;
  int K_unroll = 4;
  ir.set_order(r0, {{m, {M / M_unroll, M % M_unroll}},
                    {k, {K / K_unroll, K % K_unroll}},
                    {k, {K_unroll, 0}},
                    {m, {M_unroll, 0}}});
  // ir.set_order(r1, {{k, {K, 0}}, {n, {N, 0}}});
  // ir.set_order(r0, {{k, {K, 0}}, {m, {M, 0}}});
  ir.set_order(r1, {{m, {M / M_unroll, M % M_unroll}},
                    {n, {N / N_unroll, N % N_unroll}},
                    {k, {K / K_unroll, K % K_unroll}},
                    {k, {K_unroll, 0}},
                    {m, {M_unroll, 0}},
                    {n, {N_unroll, 0}}});
  ir.set_priority(r1, 10);
  ir.set_priority(r0, 0);
  // ir.set_order(mul, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
  // ir.set_order(add, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
  ir.set_order(mul, {{m, {M / M_unroll, M % M_unroll}},
                     {n, {N / N_unroll, N % N_unroll}},
                     {k, {K / K_unroll, K % K_unroll}},
                     {k, {K_unroll, 0}},
                     {m, {M_unroll, 0}},
                     {n, {N_unroll, 0}}});
  ir.set_order(add, {{m, {M / M_unroll, M % M_unroll}},
                     {n, {N / N_unroll, N % N_unroll}},
                     {k, {K / K_unroll, K % K_unroll}},
                     {k, {K_unroll, 0}},
                     {m, {M_unroll, 0}},
                     {n, {N_unroll, 0}}});
  ir.set_order(w, {{m, {M / M_unroll, M % M_unroll}},
                   {n, {N / N_unroll, N % N_unroll}},
                   {m, {M_unroll, 0}},
                   {n, {N_unroll, 0}}});
  ir.set_inputs({r0, r1});
  ir.set_outputs({w});
  LoopTree lt(ir);
  auto in0 = cuda_rand(M * K);
  auto in1 = cuda_rand(N * K);
  auto out = cuda_rand(M * N);
  std::cerr << lt.dump();
  std::unordered_set<LoopTree::TreeRef> threaded;
  threaded.insert(lt.roots.at(0));
  threaded.insert(lt.node(lt.roots.at(0)).children.at(0));
  std::cerr << lt.dump();
  // ca.unrolled.insert(lt.node(lt.roots.at(0)).children.at(0))
  for (auto i = 0; i < M * N; ++i) {
    out[i] = 0;
  }
  cuda_exec(lt, {in0, in1, out}, threaded);
  auto out_ref = cuda_rand(M * N);
  ref_mm(in0, in1, M, N, K, out_ref);
  float max_diff = 0;
  for (auto i = 0; i < M * N; ++i) {
    max_diff = std::max(max_diff, std::abs(out_ref[i] - out[i]));
    // std::cerr << "out: " << out[i]<< " vs " << out_ref[i] << "\n";
  }
  std::cerr << "max diff " << max_diff << "\n";
}

void test_cuda_exec(int M, int N, int K) {
  IR ir;
  auto m = ir.create_var("m");
  auto n = ir.create_var("n");
  auto k = ir.create_var("k");

  auto r0 = ir.create_node("read", {}, {m, k});
  auto r1 = ir.create_node("read", {}, {k, n});

  auto mul = ir.create_node("mul", {r1, r0}, {m, k, n});
  auto add = ir.create_node("add", {mul}, {m, n});

  auto w = ir.create_node("write", {add}, {m, n});
  int N_unroll = 4;
  int M_unroll = 4;
  int K_unroll = 4;
  ir.set_order(r0, {{m, {M / M_unroll, M % M_unroll}},
                    {k, {K / K_unroll, K % K_unroll}},
                    {k, {K_unroll, 0}},
                    {m, {M_unroll, 0}}});
  // ir.set_order(r1, {{k, {K, 0}}, {n, {N, 0}}});
  // ir.set_order(r0, {{k, {K, 0}}, {m, {M, 0}}});
  ir.set_order(r1, {{m, {M / M_unroll, M % M_unroll}},
                    {n, {N / N_unroll, N % N_unroll}},
                    {k, {K / K_unroll, K % K_unroll}},
                    {k, {K_unroll, 0}},
                    {m, {M_unroll, 0}},
                    {n, {N_unroll, 0}}});
  ir.set_priority(r1, 10);
  ir.set_priority(r0, 0);
  // ir.set_order(mul, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
  // ir.set_order(add, {{m, {M, 0}}, {k, {K, 0}}, {n, {N, 0}}});
  ir.set_order(mul, {{m, {M / M_unroll, M % M_unroll}},
                     {n, {N / N_unroll, N % N_unroll}},
                     {k, {K / K_unroll, K % K_unroll}},
                     {k, {K_unroll, 0}},
                     {m, {M_unroll, 0}},
                     {n, {N_unroll, 0}}});
  ir.set_order(add, {{m, {M / M_unroll, M % M_unroll}},
                     {n, {N / N_unroll, N % N_unroll}},
                     {k, {K / K_unroll, K % K_unroll}},
                     {k, {K_unroll, 0}},
                     {m, {M_unroll, 0}},
                     {n, {N_unroll, 0}}});
  ir.set_order(w, {{m, {M / M_unroll, M % M_unroll}},
                   {n, {N / N_unroll, N % N_unroll}},
                   {m, {M_unroll, 0}},
                   {n, {N_unroll, 0}}});
  ir.set_inputs({r0, r1});
  ir.set_outputs({w});
  LoopTree lt(ir);
  auto in0 = cuda_rand(M * K);
  auto in1 = cuda_rand(N * K);
  auto out = cuda_rand(M * N);
  std::unordered_set<LoopTree::TreeRef> threaded;
  threaded.insert(lt.roots.at(0));
  threaded.insert(lt.node(lt.roots.at(0)).children.at(0));

  CompiledCuda cc(lt, threaded);

  for (auto i = 0; i < M * N; ++i) {
    out[i] = 0;
  }
  cc({in0, in1, out});
  auto out_ref = cuda_rand(M * N);
  ref_mm(in0, in1, M, N, K, out_ref);
  float max_diff = 0;
  for (auto i = 0; i < M * N; ++i) {
    max_diff = std::max(max_diff, std::abs(out_ref[i] - out[i]));
  }
  std::cerr << "max diff " << max_diff << "\n";
}

int main() {
  // test_mm(128, 128, 128);
  // test_mm(1024, 1024, 1024);
  // test_mm2(1024, 1024, 1024);
  // test_mm(12, 12, 12);
  // test_mm2(128, 128, 128);
  // test_mm2(1023, 1021, 1025);
  // test_cuda_exec(1023, 1021, 1025);
  test_cuda_exec(8, 8, 8);
  return 0;
  // CUdevice cuDevice;
  // CUcontext context;
  // CUDA_SAFE_CALL(cuInit(0));
  // CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  // CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  // size_t total_mem, free_mem;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // std::cout << "Currently " << free_mem << " bytes free of " << total_mem
  //          << std::endl;

  IR ir;
  constexpr int N = 1024 * 1024;
  constexpr int T = 1024 * 8;
  auto a = ir.create_var("a");
  auto r0 = ir.create_node("read", {}, {a});
  auto r1 = ir.create_node("read", {}, {a});
  auto add = ir.create_node("add", {r0, r1}, {a});
  auto w = ir.create_node("write", {add}, {a});
  for (auto v : {r0, r1, add, w}) {
    // ir.set_order(v, {{a, {N, 0}}});
    // ir.set_order(v, {{a, {N / 4 / 32, 0}}, {a, {32, 0}}, {a, {4, 0}}});
    // ir.set_order(v, {{a, {N / 4, 0}}, {a, {4, 0}}});
    ir.set_order(v, {{a, {N / 4 / T, 0}}, {a, {T, 0}}, {a, {4, 0}}});
    // ir.set_order(v, {{a, {4, 0}}, {a, {N / 4, 0}}});
    ir.disable_reuse(v, 2);
  }
  ir.set_inputs({r0, r1});
  ir.set_outputs({w});
  LoopTree lt(ir);
  std::cerr << lt.dump();
  void *ptr = nullptr;
  auto err = cudaMallocManaged(&ptr, N);
  std::cerr << "latest ptr " << ptr << "\n";
  gpuErrchk(err);
  auto in0 = cuda_rand(N);
  auto in1 = cuda_rand(N);
  auto out = cuda_rand(N);
  std::unordered_set<LoopTree::TreeRef> threaded;
  assert(lt.roots.size() == 1);
  for (auto c : lt.node(lt.roots.at(0)).children) {
    threaded.insert(c);
    for (auto c2 : lt.node(c).children) {
      // ca.unrolled.insert(c2);
    }
    // ca.unrolled.insert(c);
  }
  // ca.unrolled.insert(lt.node(lt.roots.at(0)).children.at(2));
  // ca.threaded.insert(lt.node(lt.roots.at(0)).children.at(0));
  // ca.threaded[lt.roots.at(0)] = -1;
  // const int unroll_limit = 128;
  // lt.walk([&](LoopTree::TreeRef ref, int) {
  //  if (lt.node(ref).kind == LoopTree::LOOP) {
  //    return;
  //  }
  //  auto parent = lt.parent(ref);
  //  auto size = 1;
  //  while (parent != -1) {
  //    size *= lt.node(parent).loop.size;
  //    if (size > unroll_limit) {
  //      break;
  //    }
  //    if (!ca.threaded.count(parent)) {
  //      ca.unrolled.insert(parent);
  //    }
  //    parent = lt.parent(parent);
  //  }
  //});
  cuda_exec(lt, {in0, in1, out}, threaded);
  auto max_diff = 0.f;
  for (auto i = 0; i < N; ++i) {
    max_diff = std::max(std::abs(in0[i] + in1[i] - out[i]), max_diff);
  }
  std::cerr << "max diff " << max_diff << "\n";
  // for (auto i = 0; i < 5; ++i) {
  //  std::cerr << in0[i] << " + " << in1[i] << " = " << out[i] << "\n";
  //}
  cudaFree(in0);
  cudaFree(in1);
  cudaFree(out);
  // cuCtxDestroy(context);
  for (auto t : {128, 256, 512, 1024, 1024 * 32, 1024 * 128}) {
    for (auto v : {4, 8, 1}) {
      run(t, v);
    }
  }
}
