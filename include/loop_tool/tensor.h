/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once
#include "hardware.h"
#include <cstddef>
#include <cstdlib>

namespace loop_tool {

struct Tensor {
  Tensor(size_t N, int hardware = 0);
  Tensor() = delete;
  Tensor(const Tensor &) = delete;
  Tensor(Tensor &&) = default;
  ~Tensor();
  int hardware_id = -1;
  Memory data;
  size_t numel;
};

} // namespace loop_tool
