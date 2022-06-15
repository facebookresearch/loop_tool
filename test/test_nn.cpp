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

Tensor convolve(Tensor X, Tensor W, std::vector<Symbol> spatial_dims,
                std::vector<Symbol> window_dims, int stride = 1) {
  ASSERT(spatial_dims.size() == window_dims.size());
  std::vector<Symbol> new_spatial_dims;
  std::vector<Constraint> constraints;
  for (auto i = 0; i < spatial_dims.size(); ++i) {
    const auto& sp_dim = spatial_dims.at(i);
    const auto& w_dim = window_dims.at(i);
    Symbol new_dim(sp_dim.name() + "_x_" + w_dim.name());
    new_spatial_dims.emplace_back(new_dim);
    const auto& idx_equation = Expr(new_dim) * Expr(stride) + Expr(w_dim);
    constraints.emplace_back(std::make_pair(Expr(sp_dim), idx_equation));
  }

  std::vector<Symbol> batch_dims;
  std::vector<Symbol> reduction_dims = window_dims;
  auto W_dims = lt::to_set<Symbol, Hash>(W.shape());
  auto X_sp_dims = lt::to_set<Symbol, Hash>(spatial_dims);
  for (auto sym : X.shape()) {
    if (W_dims.count(sym)) {
      reduction_dims.emplace_back(sym);
    } else if (!X_sp_dims.count(sym)) {
      batch_dims.emplace_back(sym);
    }
  }

  batch_dims.insert(batch_dims.end(), new_spatial_dims.begin(),
                    new_spatial_dims.end());
  batch_dims.insert(batch_dims.end(), window_dims.begin(), window_dims.end());
  X = X.to(batch_dims, constraints);
  return (X * W).sum(reduction_dims);
}

Tensor maxpool(Tensor X, std::vector<Symbol> spatial_dims, int k,
               int stride = 1) {
  std::vector<Symbol> new_spatial_dims;
  std::vector<Symbol> new_window_dims;
  std::vector<Constraint> constraints;
  for (const auto& sym : spatial_dims) {
    Symbol new_spatial(sym.name() + "_p");
    Symbol new_window(sym.name() + "_k");
    new_spatial_dims.emplace_back(new_spatial);
    new_window_dims.emplace_back(new_window);
    auto idx_equation = Expr(new_spatial) * Expr(stride) + Expr(new_window);
    constraints.emplace_back(std::make_pair(Expr(sym), idx_equation));
    constraints.emplace_back(std::make_pair(Expr::size(new_window), Expr(k)));
  }
  std::vector<Symbol> batch_dims;
  auto spatial_dim_set = lt::to_set<Symbol, Hash>(spatial_dims);
  for (const auto& sym : X.shape()) {
    if (spatial_dim_set.count(sym)) {
      continue;
    }
    batch_dims.emplace_back(sym);
  }

  batch_dims.insert(batch_dims.end(), new_spatial_dims.begin(),
                    new_spatial_dims.end());
  batch_dims.insert(batch_dims.end(), new_window_dims.begin(),
                    new_window_dims.end());
  X = X.to(batch_dims, constraints);
  return X.max(new_window_dims);
}

TEST(MNIST) {
  auto conv = [&](Tensor X, Tensor W, Tensor B, int stride, int padding) {
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
    auto Y = convolve(X, W, {ih, iw}, {kh, kw}, stride);
    Y = Y.transpose({2, 0, 1});
    if (B.shape().size()) {
      Y = Y + B.as(Y.shape().at(0));
    }
    return Y;
  };

  auto maxp = [&](Tensor X, int k, int stride) {
    auto s = X.shape();
    return maxpool(X, {s[1], s[2]}, k, stride);
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
  }
};
