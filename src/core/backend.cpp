/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "backend.h"

std::vector<CodeGenerator> &getBackends() {
  static std::vector<CodeGenerator> backends;
  return backends;
}

std::vector<AuxCalculator> &getBackendsAux() {
  static std::vector<AuxCalculator> backends;
  return backends;
}
