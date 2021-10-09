/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "error.h"

namespace loop_tool {

struct Memory {
  int compatible = 0;
  void *address = 0;
};

struct Hardware {
  std::string name_;
  int count_;
  int id_ = 0;  // default for CPU

  Hardware(std::string name, int count) : name_(name), count_(count) {}

  void setId(int id) {
    id_ = id;
    ASSERT(id >= 0 && id < 32) << "Invalid ID for hardware: " << id;
  }

  // Allocation must be compatible with CPU
  virtual Memory alloc(size_t size) = 0;
  virtual void free(Memory &data) = 0;

  // TODO
  // virtual Memory copy(const Memory& data) = 0;
  // virtual Memory move(const Memory& data) = 0;

  bool compatible(const Memory &m) const { return m.compatible & (1 << id_); };
  const std::string &name() const { return name_; }
  int id() const { return id_; }
  int count() const { return count_; }
};

int availableCPUs();

struct CPUHardware : public Hardware {
  CPUHardware() : Hardware("cpu", availableCPUs()) {}

  Memory alloc(size_t size) override { return Memory{0x1, malloc(size)}; }

  void free(Memory &data) override {
    ::free(data.address);
    data.address = nullptr;
    data.compatible = 0;
  }

  static Hardware *create() { return new CPUHardware(); }
};

const std::vector<std::shared_ptr<Hardware>> &getHardware();
int getAvailableHardware();

int &getDefaultHardwareId();
void setDefaultHardwareId(int id);
const std::shared_ptr<Hardware> &getDefaultHardware();

void registerHardware(std::shared_ptr<Hardware> hw);

struct RegisterHardware {
  RegisterHardware(std::shared_ptr<Hardware> hw) { registerHardware(hw); }
};

}  // namespace loop_tool
