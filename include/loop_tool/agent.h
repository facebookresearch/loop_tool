/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <bitset>
#include <map>

#include "loop_tool/compile.h"


namespace loop_tool {

class LoopTreeAgent {
  enum { INST_NODE = 0, LOOP_NODE = 1, DATA_NODE = 2, CONTROL_EDGE = 3, DATA_EDGE = 4 };

public:
  LoopTree lt;
  LoopTree::TreeRef cursor;
  loop_tool::Compiler compiler;

  typedef LoopTreeAgent& (LoopTreeAgent::*ActionFn)(
      void);
  const std::map<std::string, ActionFn> actions_fn = {
      {"dummy", &LoopTreeAgent::dummy},
      {"up", &LoopTreeAgent::up},
      {"down", &LoopTreeAgent::down},
      {"swap_down", &LoopTreeAgent::swap_down},
      {"swap_up", &LoopTreeAgent::swap_up},
      // {"split_2", &LoopTreeAgent::split_2},
      // {"split_4", &LoopTreeAgent::split_4},
      // {"split_8", &LoopTreeAgent::split_8},
      // {"split_16", &LoopTreeAgent::split_16},
      // {"split_32", &LoopTreeAgent::split_32},
      // {"split_64", &LoopTreeAgent::split_64},
      // {"split_128", &LoopTreeAgent::split_128},
      // {"split_256", &LoopTreeAgent::split_256},
      // {"merge", &LoopTreeAgent::merge},
      // {"vectorize", &LoopTreeAgent::vectorize},
      // {"unroll", &LoopTreeAgent::unroll},
      // {"copy_input_0", &LoopTreeAgent::copy_input_0},
      // {"copy_input_1", &LoopTreeAgent::copy_input_1},
      // {"increase_reuse", &LoopTreeAgent::increase_reuse},
      // {"decrease_reuse", &LoopTreeAgent::decrease_reuse},
  };

  typedef double (LoopTreeAgent::*EvalFn)(void);
  const std::map<std::string, EvalFn> metrics_fn = {
      {"FLOPs", &LoopTreeAgent::FLOPs},
      {"FLOPS", &LoopTreeAgent::FLOPS},
      {"seconds", &LoopTreeAgent::seconds},
    };

  LoopTreeAgent(const LoopTree& lt, LoopTree::TreeRef cursor = 0);
  LoopTreeAgent(const LoopTreeAgent& agent);
  ~LoopTreeAgent();

  /**********************************************
   * Public API
   **********************************************/
  LoopTreeAgent& apply_action(std::string action);

  double eval(std::string metric);


  std::vector<std::string> get_available_actions();
  std::string dump();
   std::vector<std::vector<int>> get_loops_tensor() const;
  std::string dump_dot_tree_core(LoopTree::TreeRef root_tr=0) const;
  std::string dump_dot_tree() const;
  std::string create_data_node(LoopTree::TreeRef tr, IR::NodeRef nr)const;
  std::string create_lt_edge(
    std::string node_from,
    std::string node_to, 
    std::string color, 
    std::string label, 
    std::map<std::string, int> feature_dict
    )const;
  std::string create_lt_node(
    std::string node_str, 
    std::string shape, 
    std::string label, 
    std::map<std::string, int> feature_dict
    )const;
  std::string dump_dot_graph() const;
  std::string dump_dot() const;


  /**********************************************
   * Actions
   **********************************************/
  LoopTreeAgent& dummy();
  LoopTreeAgent& up();
  LoopTreeAgent& down();
  LoopTreeAgent& swap_up();
  LoopTreeAgent& swap_down();
  LoopTreeAgent& split_2();
  LoopTreeAgent& split_4();
  LoopTreeAgent& split_8();
  LoopTreeAgent& split_16();
  LoopTreeAgent& split_32();
  LoopTreeAgent& split_64();
  LoopTreeAgent& split_128();
  LoopTreeAgent& split_256();
  LoopTreeAgent& merge();
  LoopTreeAgent& vectorize();
  LoopTreeAgent& unroll();
  LoopTreeAgent& copy_input_0();
  LoopTreeAgent& copy_input_1();
  LoopTreeAgent& increase_reuse();
  LoopTreeAgent& decrease_reuse();

  /**********************************************
   * Evaluate metric
   **********************************************/

  double FLOPS();
  double FLOPs();
  double seconds();

  /**********************************************
   * Auxilary Functions
   **********************************************/
  std::string help_actions();
  std::string help_metrics();
  LoopTreeAgent& annotate(std::string annotation);
  LoopTreeAgent& split(int split_size);
};

}  // namespace loop_tool
