/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "loop_tool/compile.h"

namespace loop_tool {

class CppCompiler : public Compiler {
 public:
  CppCompiler(const LoopTree& lt);
  inline std::string gen_string() const override {
    return gen_string_impl();
  }
 private:
  std::string gen_string_impl(
      LoopTree::TreeRef ref = -1,
      std::unordered_map<IR::VarRef, int> overrides = {}) const;
  bool is_input_output(IR::NodeRef nr) const;
  std::string gen_access_string(IR::NodeRef node_ref,
                                LoopTree::TreeRef ref) const;
  std::string gen_reset_string(LoopTree::TreeRef ref) const;
  std::string gen_mem_node_string(LoopTree::TreeRef ref) const;
  std::string gen_compute_node_string(LoopTree::TreeRef ref) const;
  inline std::string gen_indent(LoopTree::TreeRef ref, int extra = 0) const {
    auto depth = ((ref == -1) ? 0 : lt.depth(ref) + 1);
    return std::string((depth + extra) * 2, ' ');
  }
  std::string gen_node_string(LoopTree::TreeRef ref) const;
  std::string gen_loop_string(
      LoopTree::TreeRef ref,
      std::unordered_map<IR::VarRef, int> overrides) const;

};

struct CppBackend : public Backend {
  CppBackend() : Backend("cpp") {}
  ~CppBackend() {}
  CppBackend(std::string name) : Backend(name) {}

  std::unique_ptr<Compiled> compile_impl(const LoopTree &lt) override;
  int hardware_requirement() const override;
};

}  // namespace loop_tool
