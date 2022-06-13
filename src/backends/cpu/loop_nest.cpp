/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "dabun/loop_nest.hpp"

#include "dabun/arithmetic_operation.hpp"
#include "dabun/code_generator/memory_resource.hpp"
#include "dabun/isa.hpp"
#include "loop_tool/backend.h"
#include "loop_tool/mutate.h"

using namespace loop_tool;
using namespace symbolic;

struct LoopNestCompiler : public Compiler {
  bool fma_nest = false;
  bool transpose_nest = false;
  LoopNestCompiler(const LoopTree& lt) : Compiler(lt) {
    fma_nest = is_fma_nest();
    transpose_nest = is_transpose_nest();
  }

  bool can_compile() const { return fma_nest || transpose_nest; }

  dabun::shared_aot_fn<void(float*, const float*, const float*, int)> gen_exec()
      const {
    ASSERT(fma_nest);
    return compile_fma_nest();
  }

#define REQUIRE(x)                    \
  {                                   \
    if (!(x)) {                       \
      std::cerr << #x << " failed\n"; \
      return false;                   \
    }                                 \
  }
  bool is_fma_nest() const {
    std::cerr << dot(lt.ir) << "\n";
    auto reads = find(lt.ir, Operation::read);
    REQUIRE(reads.size() == 2);
    REQUIRE(lt.scheduled.count(reads.at(0)) == 0);
    REQUIRE(lt.scheduled.count(reads.at(1)) == 0);

    // find mul and add operations
    auto muls = find(lt.ir, Operation::multiply);
    REQUIRE(muls.size() == 1);
    auto mul_ref = muls.at(0);
    const auto& mul = lt.ir.node(mul_ref);
    REQUIRE(lt.scheduled.count(mul_ref) == 1);
    REQUIRE(mul.inputs().size() == 2);

    auto views = find(lt.ir, Operation::view);
    for (auto v : views) {
      REQUIRE(lt.ir.node(v).outputs().size() == 1);
      REQUIRE(lt.ir.node(v).outputs().at(0) == mul_ref);
      REQUIRE(lt.ir.node(v).inputs().size() == 1);
    }

    // 2 reads (unscheduled), 1 mul, 1 add, 1 write scheduled, optional views
    REQUIRE(lt.ir.nodes().size() == 2 + 1 + 1 + 1 + views.size());

    auto adds = find(lt.ir, Operation::add);
    REQUIRE(adds.size() == 1);
    auto add_ref = adds.at(0);
    const auto& add = lt.ir.node(add_ref);
    REQUIRE(add.inputs().size() == 1);
    REQUIRE(lt.scheduled.count(add_ref) == 1);

    REQUIRE(add.inputs().size() == 1);
    REQUIRE(add.inputs().at(0) == mul_ref);

    auto mul_parent = lt.parent(lt.scheduled.at(mul_ref));
    auto add_parent = lt.parent(lt.scheduled.at(add_ref));
    REQUIRE(mul_parent == add_parent);

    auto writes = find(lt.ir, Operation::write);
    REQUIRE(writes.size() == 1);
    auto write_ref = writes.at(0);
    const auto& write = lt.ir.node(write_ref);
    REQUIRE(lt.scheduled.count(write_ref) == 1);

    return true;
  }

  bool is_transpose_nest() const { return false; }
#undef REQUIRE

  dabun::shared_aot_fn<void(float*, const float*, const float*, int)>
  compile_fma_nest() const {
    auto muls = find(lt.ir, Operation::multiply);
    const auto& mul = lt.ir.node(muls.at(0));
    auto ref = lt.parent(lt.scheduled.at(muls.at(0)));
    std::vector<std::pair<std::string, int>> order;
    std::vector<std::pair<std::string, int>> sizes;
    for (auto v : lt.ir.vars()) {
      auto size = var_sizes.at(v);
      auto size_name = lt.ir.var(v).name();
      sizes.emplace_back(size_name, (int)size);
    }

    while (ref != -1) {
      auto loop = lt.loop(ref);
      auto order_size = inner_sizes.at(ref);
      auto order_name = lt.ir.var(loop.var).name();
      order.emplace(order.begin(), order_name, order_size);
      ref = lt.parent(ref);
    }

    auto reads = find(lt.ir, Operation::read);
    auto A_ref = mul.inputs().at(0);
    const auto& A = lt.ir.node(A_ref);
    auto B_ref = mul.inputs().at(1);
    const auto& B = lt.ir.node(B_ref);
    auto C_ref = find(lt.ir, Operation::write).at(0);
    const auto& C = lt.ir.node(C_ref);

    // for strides, get the Access when reading from A, B
    auto mul_ref = lt.scheduled.at(muls.at(0));
    auto A_acc = gen_access(A_ref, mul_ref);
    auto A_idx = get_scoped_expr(A_acc);
    auto B_acc = gen_access(B_ref, mul_ref);
    auto B_idx = get_scoped_expr(B_acc);
    auto C_acc = gen_access(C_ref, lt.scheduled.at(C_ref));
    auto C_idx = get_scoped_expr(C_acc);

    std::vector<std::pair<std::string, int>> A_strides;
    std::vector<std::pair<std::string, int>> B_strides;
    std::vector<std::pair<std::string, int>> C_strides;
    for (auto v : lt.ir.vars()) {
      auto v_name = lt.ir.var(v).name();
      auto A_v = differentiate(A_idx, var_to_sym.at(v)).evaluate();
      auto B_v = differentiate(B_idx, var_to_sym.at(v)).evaluate();
      auto C_v = differentiate(C_idx, var_to_sym.at(v)).evaluate();
      if (A_v) {
        A_strides.emplace_back(v_name, A_v);
      }
      if (B_v) {
        B_strides.emplace_back(v_name, B_v);
      }
      if (C_v) {
        C_strides.emplace_back(v_name, C_v);
      }
    }

    std::vector<std::string> A_axes;
    std::vector<std::string> B_axes;
    std::vector<std::string> C_axes;
    for (auto v : lt.ir.node(A_ref).vars()) {
      A_axes.emplace_back(lt.ir.var(v).name());
    }
    for (auto v : lt.ir.node(B_ref).vars()) {
      B_axes.emplace_back(lt.ir.var(v).name());
    }
    for (auto v : lt.ir.node(C_ref).vars()) {
      C_axes.emplace_back(lt.ir.var(v).name());
    }

    auto arg = dabun::LN_sizes(sizes)
                   .C_axes(C_axes)
                   .A_axes(A_axes)
                   .B_axes(B_axes)
                   .C_strides(C_strides)
                   .A_strides(A_strides)
                   .B_strides(B_strides)
                   .append_loops(order);

#if defined(__AVX512F__)
#define VEX dabun::extension::avx512
#elif defined(__aarch64__) || defined(__arm64__)
#define VEX dabun::extension::neon
#else  // default to avx2
#define VEX dabun::extension::avx2
#endif

    return dabun::loop_nest_compiler<VEX, float>(arg, dabun::fma).get_shared();
  }

  void compile_transpose_nest() const {
    ASSERT(0);
    return;
  }
};

struct LoopNestCompiled : public Compiled {
  dabun::shared_aot_fn<void(float*, const float*, const float*, int)> fn;

  LoopNestCompiled() = delete;
  LoopNestCompiled(const LoopNestCompiled&) = delete;
  LoopNestCompiled(LoopNestCompiled&&) = delete;

  LoopNestCompiled(const LoopTree& lt) {
    LoopNestCompiler cc(lt);
    fn = std::move(cc.gen_exec());
  }

  ~LoopNestCompiled() {}

  void run(const std::vector<void*>& memory, bool sync) const override {
    fn((float*)(memory[2]), (const float*)(memory[0]),
       (const float*)(memory[1]), 0);
  }

  std::string dump() const override { return ""; }
};

struct LoopNestBackend : public Backend {
  LoopNestBackend() : Backend("loop_nest") {
    // static destruction order hack
    (void)dabun::memory_resource::default_resource();
  }
  ~LoopNestBackend() {}

  std::unique_ptr<Compiled> compile_impl(const LoopTree& lt) const override {
    return std::make_unique<LoopNestCompiled>(lt);
  }
  int hardware_requirement() const override { return 1 << 0; }
};

static RegisterBackend loop_nest_backend_reg_(
    std::make_shared<LoopNestBackend>());
