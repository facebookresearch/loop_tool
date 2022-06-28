/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <loop_tool/loop_tool.h>

namespace loop_tool {

std::string serialize(const IR& ir);
IR deserialize(const std::string& str);

std::string serialize_looptree(const LoopTree looptree);
LoopTree deserialize_looptree(const std::string str);

std::string serialize_looptree_agent(const LoopTreeAgent agent);
LoopTreeAgent deserialize_looptree_agent(const std::string str);

}  // namespace loop_tool
