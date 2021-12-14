/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once
#include <loop_tool/loop_tool.h>

namespace loop_tool {

LoopTree split(const LoopTree& lt, LoopTree::TreeRef ref, int64_t size);
LoopTree swap(const LoopTree& lt, LoopTree::TreeRef a, LoopTree::TreeRef b);

}  // namespace loop_tool
