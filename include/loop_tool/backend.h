/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "ir.h"
#include "tensor.h"

namespace loop_tool {

struct Compiled {
  virtual ~Compiled() {}
  virtual void run(const std::vector<void *> &memory,
                   bool sync = true) const = 0;

  void operator()(const std::vector<Tensor *> &tensors, bool sync = true) const;

  template <bool sync, typename... Args>
  void run(Args const &... tensors) const {
    std::vector<void *> memory = {tensors.data.address...};
    run(memory, sync);
  }

  template <typename... Args>
  void operator()(Args const &... tensors) const {
    run<true, Args...>(std::forward<Args const &>(tensors)...);
  }

  template <typename... Args>
  void async(Args const &... tensors) const {
    run<false, Args...>(std::forward<Args const &>(tensors)...);
  }

  std::unordered_map<std::string, int> int_properties;
  std::unordered_map<std::string, std::string> string_properties;
  int hardware_requirement = -1;
  std::string name;
};

struct Backend {
  std::string name_;
  Backend(std::string name) : name_(name) {}
  virtual ~Backend(){};

  const std::string &name() const { return name_; }

  virtual std::unique_ptr<Compiled> compile_impl(
      const LoopTree &lt, const std::unordered_set<LoopTree::TreeRef> &parallel,
      LoopTree::TreeRef root) = 0;
  virtual int hardware_requirement() const = 0;
  std::unique_ptr<Compiled> compile(
      const LoopTree &lt, const std::unordered_set<LoopTree::TreeRef> &parallel,
      LoopTree::TreeRef root) {
    auto compiled = compile_impl(lt, parallel, root);
    compiled->hardware_requirement = hardware_requirement();
    compiled->name = name();
    return compiled;
  }
};

const std::unordered_map<std::string, std::shared_ptr<Backend>> &getBackends();
void registerBackend(std::shared_ptr<Backend> backend);

struct RegisterBackend {
  RegisterBackend(std::shared_ptr<Backend> backend) {
    registerBackend(backend);
  }
};

}  // namespace loop_tool
