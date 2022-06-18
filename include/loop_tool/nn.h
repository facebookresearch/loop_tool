/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <loop_tool/loop_tool.h>

namespace loop_tool {
namespace nn {

lazy::Tensor convolve(lazy::Tensor X, lazy::Tensor W,
                      std::vector<symbolic::Symbol> spatial_dims,
                      std::vector<symbolic::Symbol> window_dims,
                      int stride = 1);
lazy::Tensor maxpool(lazy::Tensor X, std::vector<symbolic::Symbol> spatial_dims,
                     int k, int stride = 1);

}  // namespace nn
}  // namespace loop_tool
