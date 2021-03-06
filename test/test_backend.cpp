/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>

#include "test_utils.h"

namespace lt = loop_tool;

struct CustomCompiled : public lt::Compiled {
  void run(const std::vector<void *> &memory, bool sync) const override {
    std::cerr << "here!\n";
    return;
  }
};

struct CustomBackend : lt::Backend {
  CustomBackend() : lt::Backend("custom") {}

  std::unique_ptr<lt::Compiled> compile_impl(
      const lt::LoopTree &lt) const override {
    return std::make_unique<CustomCompiled>();
  }

  int hardware_requirement() const override {
    return 0;  // CPU
  }
};

static lt::RegisterBackend custom_backend_reg_{
    std::make_shared<CustomBackend>()};

TEST(CustomBackend) {
  // define
  lt::IR ir;
  auto a = ir.create_var("a");
  auto b = ir.create_var("b");
  auto r = ir.create_node(lt::Operation::read, {}, {a, b});
  auto add = ir.create_node(lt::Operation::add, {r}, {});
  auto w = ir.create_node(lt::Operation::write, {add}, {});
  ir.set_inputs({r});
  ir.set_outputs({w});

  // schedule
  constexpr int N = 16;
  /*
     read and add nodes have the loop order:

     ```
     for a in N:
       for b in N:
         read
         add
     ```

   **/
  ir.set_order(r, {{a, {N, 0}}, {b, {N, 0}}});
  ir.set_order(add, {{a, {N, 0}}, {b, {N, 0}}});
  // write can be executed without looping
  ir.set_order(w, {});
  lt::LoopTree loop_tree(ir);

  std::cout << loop_tree.dump();

  // compile and run
  auto compiled = lt::getBackends().at("custom")->compile(loop_tree);
  auto A = lt::Tensor(N * N);
  auto B = lt::Tensor(1);
  const auto &f = *compiled;
  f(A, B);
  f.async(A, B);
}
