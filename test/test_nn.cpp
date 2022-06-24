/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <loop_tool/loop_tool.h>
#include <string.h>

#include <chrono>
#include <random>

#include "test_utils.h"

namespace lt = loop_tool;
using namespace loop_tool::lazy;
using namespace loop_tool::symbolic;
using namespace loop_tool::testing;

TEST(MNIST) {
  loop_tool::ScopedBackend sb("wasm");
  auto conv = [](Tensor X, Tensor W, Tensor B, int stride, int padding) {
    auto inp_shape = X.shape();
    if (padding > 0) {
      // H
      X = X.pad(inp_shape[1], padding);
      // W
      X = X.pad(inp_shape[2], padding);
    }
    auto w_shape = W.shape();
    auto inp_padded_shape = X.shape();
    auto oc = w_shape[0];
    auto ic = w_shape[1];
    auto kh = w_shape[2];
    auto kw = w_shape[3];
    auto ih = inp_padded_shape[1];
    auto iw = inp_padded_shape[2];
    X = X.as(ic, ih, iw);
    auto Y = lt::nn::convolve(X, W, {ih, iw}, {kh, kw}, stride);
    Y = Y.transpose({2, 0, 1});
    if (B.shape().size()) {
      Y = Y + B.as(Y.shape().at(0));
    }
    return Y;
  };

  auto maxp = [](Tensor X, int k, int stride) {
    auto s = X.shape();
    return lt::nn::maxpool(X, {s[1], s[2]}, k, stride);
  };
  for (auto i = 0; i < 100; ++i) {
    Tensor X(1, 28, 28);
    Tensor W0(16, 1, 5, 5);
    Tensor B0(16);
    Tensor W1(32, 16, 5, 5);
    Tensor B1(32);
    X = conv(X, W0, B0, 1, 2);
    X = maxp(X, 2, 2);
    X = conv(X, W1, B1, 1, 2);
    X = maxp(X, 2, 2);
    //(void)X.sizes()[0];
    X.compile();
    X.clear_cache();
    std::cerr << "hash is " << X.hash() << "\n";
  }
};
