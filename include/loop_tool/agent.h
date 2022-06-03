/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <map>

#include "ir.h"
#include "mutate.h"

namespace loop_tool {

class LoopTreeAgent {
 public:
  LoopTreeAgent(const LoopTree& lt, LoopTree::TreeRef cursor = 0)
      : lt(lt), cursor(cursor) {}

  LoopTreeAgent(LoopTreeAgent& agent) : lt(agent.lt), cursor(agent.cursor) {}

  ~LoopTreeAgent() {}

  /**********************************************
   * Public API
   **********************************************/
  LoopTreeAgent& apply_action(std::string action) {
    if (actions_fn.count(action) > 0) {
      std::invoke(actions_fn.at(action), this);
    } else {
      std::cout << "Action: " << action << " not available" << std::endl;
    }
    return *this;
  }

  std::vector<std::string> get_available_actions() {
    LoopTree lt_copy(lt);
    LoopTree::TreeRef cursor_copy(cursor);

    std::vector<std::string> available_actions;

    for (auto& action : actions_fn) {
      try {
        apply_action(action.first);
        available_actions.push_back(action.first);
        // std::cout << action.first << std::endl;
      } catch (std::runtime_error e) {
        // std::cout << "Action: " << action.first << " illegal" << std::endl;
      }
      lt = lt_copy;
      cursor = cursor_copy;
    }

    return available_actions;
  }

  std::string dump() {
    return lt.dump([=](LoopTree::TreeRef ref) -> std::string {
      if (ref == cursor) {
        return "<<<<<< cursor (line " + std::to_string(cursor) + " )";
      }
      return "";
    });
  }

 private:
  LoopTree lt;
  LoopTree::TreeRef cursor;
  typedef LoopTreeAgent& (LoopTreeAgent::*ActionFn)(
      void);  // function pointer type
  const std::map<std::string, ActionFn> actions_fn = {
      {"up", &LoopTreeAgent::up},
      {"down", &LoopTreeAgent::down},
      {"swap_down", &LoopTreeAgent::swap_down},
      {"swap_up", &LoopTreeAgent::swap_up},
      {"split_2", &LoopTreeAgent::split_2},
      {"split_4", &LoopTreeAgent::split_4},
      {"split_8", &LoopTreeAgent::split_8},
      {"split_16", &LoopTreeAgent::split_16},
      {"split_32", &LoopTreeAgent::split_32},
      {"split_64", &LoopTreeAgent::split_64},
      {"split_128", &LoopTreeAgent::split_128},
      {"split_256", &LoopTreeAgent::split_256},
      {"merge", &LoopTreeAgent::merge},
      {"vrctorize", &LoopTreeAgent::vectorize},
      {"unroll", &LoopTreeAgent::unroll},
      {"copy_input_0", &LoopTreeAgent::copy_input_0},
      {"copy_input_1", &LoopTreeAgent::copy_input_1},
      {"increase_reuse", &LoopTreeAgent::increase_reuse},
      {"decrease_reuse", &LoopTreeAgent::decrease_reuse},
  };

  /**********************************************
   * Actions
   **********************************************/
  LoopTreeAgent& up() {
    cursor = loop_tool::previous_ref(lt, cursor);
    return *this;
  }

  LoopTreeAgent& down() {
    cursor = loop_tool::next_ref(lt, cursor);
    return *this;
  }

  LoopTreeAgent& swap_up() {
    lt = loop_tool::try_swap(lt, cursor, previous_ref(lt, cursor));
    return *this;
  }

  LoopTreeAgent& swap_down() {
    lt = loop_tool::try_swap(lt, cursor, next_ref(lt, cursor));
    return *this;
  }

  LoopTreeAgent& split_2() { return split(2); }

  LoopTreeAgent& split_4() { return split(4); }

  LoopTreeAgent& split_8() { return split(8); }

  LoopTreeAgent& split_16() { return split(16); }

  LoopTreeAgent& split_32() { return split(32); }

  LoopTreeAgent& split_64() { return split(64); }

  LoopTreeAgent& split_128() { return split(128); }

  LoopTreeAgent& split_256() { return split(256); }

  LoopTreeAgent& merge() {
    lt = loop_tool::merge(lt, cursor);
    return *this;
  }

  LoopTreeAgent& vectorize() { return annotate("vectorize"); }

  LoopTreeAgent& unroll() { return annotate("unroll"); }

  LoopTreeAgent& copy_input_0() {
    auto input_id = loop_tool::get_inputs(lt, cursor)[0];
    lt = loop_tool::copy_input(lt, cursor, input_id);
    return *this;
  }

  LoopTreeAgent& copy_input_1() {
    auto input_id = loop_tool::get_inputs(lt, cursor)[1];
    lt = loop_tool::copy_input(lt, cursor, input_id);
    return *this;
  }

  LoopTreeAgent& increase_reuse() {
    lt = loop_tool::increase_reuse(lt, cursor);
    return *this;
  }

  LoopTreeAgent& decrease_reuse() {
    lt = loop_tool::decrease_reuse(lt, cursor);
    return *this;
  }

  /**********************************************
   * Auxilary Actions
   **********************************************/
  LoopTreeAgent& annotate(std::string annotation) {
    if (lt.annotation(cursor) == annotation) {
      lt = loop_tool::annotate(lt, cursor, "");
    } else {
      lt = loop_tool::annotate(lt, cursor, annotation);
    }
    return *this;
  }

  LoopTreeAgent& split(int split_size) {
    lt = loop_tool::split(lt, cursor, split_size);
    return *this;
  }
};

}  // namespace loop_tool
