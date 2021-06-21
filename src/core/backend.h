/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "compile.h"

using CodeGenerator = std::function<InnerFnType(
    const LoopTree &lt, const Auxiliary &aux, LoopTree::TreeRef ref)>;
using AuxCalculator = std::function<bool(const LoopTree &lt, Auxiliary &aux,
                                         LoopTree::TreeRef ref)>;

std::vector<CodeGenerator> &getBackends();
std::vector<AuxCalculator> &getBackendsAux();

struct RegisteredBackend {
  RegisteredBackend(const CodeGenerator &fn, const AuxCalculator &aux_fn) {
    auto &backends = getBackends();
    backends.emplace_back(fn);
    auto &backends_aux = getBackendsAux();
    backends_aux.emplace_back(aux_fn);
  }
};

#define REGISTER_BACKEND(backend, fn, aux_fn)                                  \
  static RegisteredBackend _registered_backend_##backend_##fn(fn, aux_fn);
