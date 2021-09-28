/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/backend.h"

#include <mutex>
#include <unordered_map>

static std::mutex registration_mutex_;

namespace loop_tool {

void Compiled::operator()(const std::vector<Tensor *> &tensors,
                          bool sync) const {
  std::vector<void *> memory;
  for (const auto &t : tensors) {
    memory.emplace_back(t->data.address);
  }
  run(memory, sync);
}

std::unordered_map<std::string, std::shared_ptr<Backend>>
    &getMutableBackends() {
  static std::unordered_map<std::string, std::shared_ptr<Backend>> backends_;
  return backends_;
}

const std::unordered_map<std::string, std::shared_ptr<Backend>> &getBackends() {
  return getMutableBackends();
}

void registerBackend(std::shared_ptr<Backend> backend) {
  std::lock_guard<std::mutex> guard(registration_mutex_);
  getMutableBackends()[backend->name()] = backend;
}

std::shared_ptr<Backend> &getDefaultBackend() {
  static std::shared_ptr<Backend> default_backend_;
  if (!default_backend_) {
    ASSERT(getBackends().count("cpu")) << "couldn't find cpu (default) backend";
    default_backend_ = getBackends().at("cpu");
  }
  return default_backend_;
}

void setDefaultBackend(std::string backend) {
  ASSERT(getBackends().count(backend)) << "couldn't find backend " << backend;
  getDefaultBackend() = getBackends().at(backend);
}

}  // namespace loop_tool
