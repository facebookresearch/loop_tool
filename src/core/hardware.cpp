/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/hardware.h"

#include <iostream>
#include <mutex>

static std::mutex registration_mutex_;

namespace loop_tool {

std::vector<std::shared_ptr<Hardware>> &getMutableHardware() {
  // We want CPU to be first, so we don't use registration pattern
  static std::vector<std::shared_ptr<Hardware>> hardware_ = {
      std::make_shared<CPUHardware>()};
  return hardware_;
}
const std::vector<std::shared_ptr<Hardware>> &getHardware() {
  return getMutableHardware();
}

int getAvailableHardware() {
  int avail = 0;
  for (auto &hw : getHardware()) {
    if (hw->count()) {
      avail |= 1 << hw->id();
    }
  }
  return avail;
}

void registerHardware(std::shared_ptr<Hardware> hw) {
  std::lock_guard<std::mutex> guard(registration_mutex_);
  hw->setId(getHardware().size());
  getMutableHardware().emplace_back(hw);
}

int availableCPUs() {
  // TODO
  return 1;
}

}  // namespace loop_tool
