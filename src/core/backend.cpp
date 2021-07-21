/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "backend.h"
#include <mutex>
#include <unordered_map>

static std::mutex registration_mutex_;

std::unordered_map<std::string, std::shared_ptr<Backend>> &
getMutableBackends() {
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
