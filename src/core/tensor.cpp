/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/tensor.h"
#include <iostream>

using namespace loop_tool;

Tensor::Tensor(size_t N, int hardware) : hardware_id(hardware) {
  data = getHardware().at(hardware_id)->alloc(N * sizeof(float));
  numel = N;
}

Tensor::~Tensor() { getHardware().at(hardware_id)->free(data); }
