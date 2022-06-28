/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/


#include <bitset>
#include <map>

#include "loop_tool/agent.h"
#include "loop_tool/ir.h"
#include "loop_tool/mutate.h"
// #include "serialization.h"
#include "loop_tool/compile.h"


namespace loop_tool {
  LoopTreeAgent::LoopTreeAgent(const LoopTree& lt, LoopTree::TreeRef cursor)
    : lt(lt), cursor(cursor), compiler(loop_tool::Compiler(lt)) {}

  LoopTreeAgent::LoopTreeAgent(const LoopTreeAgent& agent)
      :lt(agent.lt), cursor(cursor), compiler(loop_tool::Compiler(agent.lt)) {}

  LoopTreeAgent::~LoopTreeAgent(){}

/**********************************************
   * Public API
   **********************************************/
  LoopTreeAgent& LoopTreeAgent::apply_action(std::string action) {
    ASSERT(actions_fn.count(action) > 0) << help_actions();
    std::invoke(actions_fn.at(action), this);
    return *this;
  }

  double LoopTreeAgent::eval(std::string metric){
    ASSERT(metrics_fn.count(metric) > 0) << help_metrics();
    return std::invoke(metrics_fn.at(metric), this);
  }


  std::vector<std::string> LoopTreeAgent::get_available_actions() {
    LoopTree lt_copy(lt);
    LoopTree::TreeRef cursor_copy(cursor);

    std::vector<std::string> available_actions;

    for (auto& action : actions_fn) {
      try {
        // std::cout << "Action: "<< action.first << std::endl;
        apply_action(action.first);
        // std::cout << dump();
        FLOPS();
        // eval_runtime(lt);
        available_actions.push_back(action.first);
      } catch (std::exception& e) {
        // std::cout << "Action: "<< action.first << " Error:: " << e.what() << std::endl;
      }
      lt = lt_copy;
      cursor = cursor_copy;
    }

    return available_actions;
  }


  std::string LoopTreeAgent::dump() {
    return lt.dump([=](LoopTree::TreeRef ref) -> std::string {
      if (ref == cursor) {
        return "<<<<<< cursor (line " + std::to_string(cursor) + " )";
      }
      return "";
    });
  }


  std::string LoopTreeAgent::dump_tensor() const {
    int loop_id = 0;
    std::unordered_map<std::string, int> iter_id;
    std::stringstream ss;
    auto ir_coordinates = lt.get_ir_coordinates(cursor);


    ss << "[\n";
    auto short_name = [](std::string name) {
      return name.substr(0, name.find("_"));
    };
      lt.walk(
        [&](LoopTree::TreeRef tr, int depth) {
            auto tn = lt.tree_node(tr);
            if (tn.kind == LoopTree::NODE){
              return;
            }
            
            // ss << "[";
            // ss << "0" << ",";            
            ss << (tr == cursor) << ",";            
            ss << (lt.annotation(tr) == "vectorize") << ",";
            ss << (lt.annotation(tr) == "unroll") << ",";
            ss << tn.loop.var << ",";
            ss << tn.loop.size << ",";
            ss << tn.loop.tail << ",";

            // for(auto &ch: std::bitset< 4 >(  tn.loop.var ).to_string()){
            //   ss << ch << ",";
            // } 
            // for(auto &ch: std::bitset< 16 >(  tn.loop.size ).to_string()){
            //   ss << ch << ",";
            // }
            // for(auto &ch: std::bitset< 16 >(  tn.loop.tail ).to_string()){
            //   ss << ch << ",";
            // }

            // ss << "],";
          },
          0);

    ss << "]\n";
    return ss.str();
  }


  std::string LoopTreeAgent::dump_dot_tree_core(LoopTree::TreeRef root_tr) const {
    // int n = 0;
    std::stringstream ss;

    lt.walk(
      [&](LoopTree::TreeRef tr, int depth) {
          auto tn = lt.tree_node(tr);

          ss << "L" << root_tr++ << " [shape=record,";
          ss << "label=\"{";

          if (tn.kind == LoopTree::NODE){
            ss << "'type':" << LoopTree::NODE << ",";
            ss << "'vectorize':" << 0 << ",";
            ss << "'unroll':" << 0 << ",";
            ss << "'name':" << 0 << ",";
            ss << "'size':" << 0 << ",";
            ss << "'tail':" << 0 << ",";
          }else if (tn.kind == LoopTree::LOOP){
            ss << "'type':" << LoopTree::LOOP << ",";
            ss << "'vectorize':" << (lt.annotation(tr) == "vectorize") << ",";
            ss << "'unroll': " << (lt.annotation(tr) == "unroll") << ",";
            ss << "'name':" << tn.loop.var << ",";
            ss << "'size':" << tn.loop.size << ",";
            ss << "'tail':" << tn.loop.tail << ",";
          }
          ss << "'cursor':" << (tr == cursor) << ",";

          ss << "}\"];\n";

          for (auto out : lt.children(tr)) {
            ss << " " << tr << " -> " << out << ";\n";
          }
        },
        root_tr);

    return ss.str();
  } 

  std::string LoopTreeAgent::dump_dot_tree() const {
    int loop_id = 0;
    std::unordered_map<std::string, int> iter_id;
    std::stringstream ss;
    auto ir_coordinates = lt.get_ir_coordinates(cursor);


    ss << "digraph G {\n";
    ss << " node [fontname = \"courier\", fontsize=12];\n";
    auto short_name = [](std::string name) {
      return name.substr(0, name.find("_"));
    };

    ss << dump_dot_tree_core();

    ss << "}\n";
    return ss.str();
  }

  std::string LoopTreeAgent::create_data_node(LoopTree::TreeRef tr, IR::NodeRef nr)const{
    auto &node_in = lt.ir.node(nr);
    auto ir_vars = node_in.vars();

    std::stringstream ss;

    std::string node_str = "D" + std::to_string(nr);
    std::string shape = "ellipse";
    std::string label = [&]() {
            std::stringstream ss_tmp;
            ss_tmp << "%" << nr << "[ ";
            for (auto &v : ir_vars) {
              ss_tmp << lt.ir.var(v).name() << " ";
            }
            ss_tmp << "]";
            return ss_tmp.str();
        }();
    const std::map<std::string, int> feature_dict = 
      {
        {"type", DATA_NODE}
      };
    ss << create_lt_node( node_str, shape, label, feature_dict);



    // Connect the node with loop nodes that access it
    auto access = compiler.gen_access(nr, tr);
    const auto& idx_expr = compiler.get_scoped_expr(access);
    auto sym_strides = compiler.get_symbol_strides(tr, access.alloc.lca);
    std::cout << "NODE = " << tr << "\n";
    std::cout << idx_expr.dump() << '\n';

    std::cout << "^^^^^^^^^^^^^^^^^^^^^^NodeIR = " << nr << std::endl;

    for (auto &sym_stride: sym_strides){
      std::cout << sym_stride.first.name_ << " === " << sym_stride.first.id() << "\n";
      auto base_stride = differentiate(idx_expr, sym_stride.first);
      ASSERT(base_stride.can_evaluate());
      for (auto &tr_stride: sym_stride.second){

        std::cout << tr_stride.first << ", " << tr_stride.second << "\n"; 
          ss << create_lt_edge(
            "L" + std::to_string(tr_stride.first), 
            "D" + std::to_string(nr),
            "red", 
            std::to_string(tr_stride.second * base_stride.evaluate()), 
            { 
              {"type", DATA_EDGE},
              {"stride", tr_stride.second}
            });
      }
      std::cout << "+++++++++++++\n";
    }



    return ss.str();
  }

  std::string LoopTreeAgent::create_lt_edge(
    std::string node_from,
    std::string node_to, 
    std::string color, 
    std::string label, 
    std::map<std::string, int> feature_dict
    )const{
    std::stringstream ss;
    ss << node_from << " -> " << node_to << " [";
    ss << "color=\"" << color << "\",";
    ss << "label=\"" << label << "\",";
    ss << "feature_dict=\"{";
    for (const auto &item: feature_dict){
      ss << "'" << item.first << "':" << item.second << ",";
    }
    ss << "}\"];\n";
    return ss.str();
  }

  std::string LoopTreeAgent::create_lt_node(
    std::string node_str, 
    std::string shape, 
    std::string label, 
    std::map<std::string, int> feature_dict
    )const{
    std::stringstream ss;
    ss << node_str << " [shape=" << shape << ",";
    ss << "label=\"" << label << "\",";
    ss << "feature_dict=\"{";
    for (const auto &item: feature_dict){
      ss << "'" << item.first << "':" << item.second << ",";
    }
    ss << "}\"];\n";
    return ss.str();
  }

  std::string LoopTreeAgent::dump_dot_graph() const {
    int loop_id = 0;
    std::unordered_map<std::string, int> iter_id;
    std::vector<int> created_data_nodes;
    
    std::stringstream ss, ss_print;
    auto ir_coordinates = lt.get_ir_coordinates(cursor);


    ss << "digraph G {\n";
    ss << " node [fontname = \"courier\", fontsize=12];\n";
    auto short_name = [](std::string name) {
      return name.substr(0, name.find("_"));
    };

    
    int n = 0;
    lt.walk(
      [&](LoopTree::TreeRef tr, int depth) {
          auto tn = lt.tree_node(tr);

          if (tn.kind == LoopTree::NODE){
            auto &node = lt.ir.node(tn.node);

            // Create LoopTree node
            ss << create_lt_node( 
              "L" + std::to_string(tr), 
              "hexagon", 
              lt.ir.dump(tn.node), 
              {
                {"type", INST_NODE},
                {"cursor", tr == cursor}
              });

            // Create data_in nodes
            for (const auto &inp : node.inputs()) {              
              if (!std::count(created_data_nodes.begin(), created_data_nodes.end(), inp)){
                ss << create_data_node(tr, inp);
                created_data_nodes.push_back(inp);              
              } 
              // Connect current Node with data_in nodes             
              ss << create_lt_edge( 
                "D" + std::to_string(inp), 
                "L" + std::to_string(tr),
                "blue", 
                "",
                { 
                  {"type", DATA_EDGE},
                  {"stride", 1}
                });
            }

            // Create data_out node
            if (!std::count(created_data_nodes.begin(), created_data_nodes.end(), tn.node)){
              ss << create_data_node(tr, tn.node);
              created_data_nodes.push_back(tn.node);              
            }                   
            // Connect current Node with data_out nodes
            ss << create_lt_edge( 
              "L" + std::to_string(tr), 
              "D" + std::to_string(tn.node),
              "blue", 
              "", 
              {
                {"type", DATA_EDGE},
                {"stride", 1}                
              });

          }else if (tn.kind == LoopTree::LOOP){
            // Create LoopTree node
            ss << create_lt_node( 
              "L" + std::to_string(tr), 
              "record", 
              "for " + lt.ir.var(tn.loop.var).name() + " in " + std::to_string(tn.loop.size) + " r " + std::to_string(tn.loop.tail),
               {
                {"type", LOOP_NODE},
                {"cursor", tr == cursor},
                {"vectorize", (lt.annotation(tr) == "vectorize")},
                {"unroll", (lt.annotation(tr) == "unroll")},
                {"size", tn.loop.size},
                {"tail", tn.loop.tail}
              });
          }

          // Connect with children
          for (auto child : lt.children(tr)) {
            ss << create_lt_edge( 
              "L" + std::to_string(tr), 
              "L" + std::to_string(child),
              "black", 
              "", 
              {
                {"type", CONTROL_EDGE}
              });
          }

        },
        0);

    ss << "}\n";
    return ss.str();
  }


  std::string LoopTreeAgent::dump_dot() const {
    int loop_id = 0;
    std::unordered_map<std::string, int> iter_id;
    std::stringstream ss;
    auto ir_coordinates = lt.get_ir_coordinates(cursor);


    ss << "digraph G {\n";
    ss << " node [fontname = \"courier\", fontsize=12];\n";
    auto short_name = [](std::string name) {
      return name.substr(0, name.find("_"));
    };
    for (auto n : toposort(lt.ir)) {
      ss << " ";
      ss << n << "[shape=record,";
      ss << "label=\"{";
      ss << "'node':'" << loop_tool::dump(lt.ir.node(n).op()) << "',";
      ss << "'iters': [";
      auto vars = lt.ir.node(n).vars();
      for (auto &v : vars) {
        ss << "'" << short_name(lt.ir.var(v).name()) << "'";
        if (&v != &vars.back()) {
          ss << ", ";
        }
      }
      ss << "],";
      auto order = lt.ir.order(n);
      int i = 0;
      for (auto &p : order) {
        ss << "'L" << i << "':{";

        std::string iter_name = short_name(lt.ir.var(p.first).name());
        if (iter_id.count(iter_name) == 0) {
          iter_id[iter_name] = iter_id.size();
        }

        ss << "'name':" << iter_id[iter_name] << ",";

        if (p.second.size >= 0) {
          ss << "'range':" << p.second.size << ",";
        }

        if (p.second.tail >= 0) {
          ss << "'tail':" << p.second.tail << ",";
        }

        ss << "'vectorize':" << (lt.ir.loop_annotations(n)[i] == "vectorize") << ",";
        ss << "'unroll':" << (lt.ir.loop_annotations(n)[i] == "unroll") << ",";

        bool is_cursor = std::count(ir_coordinates.begin(), ir_coordinates.end(), std::make_pair(n, i));
        ss << "'cursor':" << is_cursor << ",";
        

        loop_id++;
        i++;
        ss << "},";
      }
      i = 0;

      ss << "}\"];\n";
      for (auto out : lt.ir.node(n).outputs()) {
        ss << " " << n << " -> " << out << ";\n";
      }
    }
    ss << "}\n";
    return ss.str();
  }


  /**********************************************
   * Actions
   **********************************************/
  LoopTreeAgent& LoopTreeAgent::dummy() {
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::up() {
    cursor = loop_tool::previous_ref(lt, cursor);
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::down() {
    cursor = loop_tool::next_ref(lt, cursor);
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::swap_up() {
    lt = loop_tool::try_swap(lt, cursor, previous_ref(lt, cursor));
    up();
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::swap_down() {
    lt = loop_tool::try_swap(lt, cursor, next_ref(lt, cursor));
    down();
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::split_2() { return split(2); }

  LoopTreeAgent& LoopTreeAgent::split_4() { return split(4); }

  LoopTreeAgent& LoopTreeAgent::split_8() { return split(8); }

  LoopTreeAgent& LoopTreeAgent::split_16() { return split(16); }

  LoopTreeAgent& LoopTreeAgent::split_32() { return split(32); }

  LoopTreeAgent& LoopTreeAgent::split_64() { return split(64); }

  LoopTreeAgent& LoopTreeAgent::split_128() { return split(128); }

  LoopTreeAgent& LoopTreeAgent::split_256() { return split(256); }

  LoopTreeAgent& LoopTreeAgent::merge() {
    lt = loop_tool::merge(lt, cursor);
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::vectorize() { return annotate("vectorize"); }

  LoopTreeAgent& LoopTreeAgent::unroll() { return annotate("unroll"); }

  LoopTreeAgent& LoopTreeAgent::copy_input_0() {
    auto input_id = loop_tool::get_inputs(lt, cursor)[0];
    lt = loop_tool::copy_input(lt, cursor, input_id);
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::copy_input_1() {
    auto input_id = loop_tool::get_inputs(lt, cursor)[1];
    lt = loop_tool::copy_input(lt, cursor, input_id);
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::increase_reuse() {
    lt = loop_tool::increase_reuse(lt, cursor);
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::decrease_reuse() {
    lt = loop_tool::decrease_reuse(lt, cursor);
    return *this;
  }

  /**********************************************
   * Evaluate metric
   **********************************************/

  double LoopTreeAgent::FLOPS() {
    return (double) loop_tool::FLOPS(lt);
  }

  double LoopTreeAgent::FLOPs() {
    return (double) loop_tool::FLOPs(lt);
  }

  double LoopTreeAgent::seconds() {
    return (double) loop_tool::eval_runtime(lt);
  }

  /**********************************************
   * Auxilary Functions
   **********************************************/
  std::string LoopTreeAgent::help_actions(){
    std::stringstream ss_actions;

    ss_actions << "Available actions are:" << std::endl;
    for (auto& action : actions_fn) {
      ss_actions << action.first << std::endl;
    }
    return ss_actions.str();
  }

  std::string LoopTreeAgent::help_metrics(){
    std::stringstream ss_metrics;

    ss_metrics << "Available metrics are:" << std::endl;
    for (auto& metric : metrics_fn) {
      ss_metrics << metric.first << std::endl;
    }
    return ss_metrics.str();
  }

  LoopTreeAgent& LoopTreeAgent::annotate(std::string annotation) {
    if (lt.annotation(cursor) == annotation) {
      lt = loop_tool::annotate(lt, cursor, "");
    } else {
      lt = loop_tool::annotate(lt, cursor, annotation);
    }
    return *this;
  }

  LoopTreeAgent& LoopTreeAgent::split(int split_size) {
    lt = loop_tool::split(lt, cursor, split_size);
    return *this;
  }

}  // namespace loop_tool