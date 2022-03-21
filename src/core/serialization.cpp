#include "loop_tool/serialization.h"

namespace loop_tool {

// vars
//  name
// nodes
//  op
//  inputs
//  vars
//  constraints
//  sym->var map
//  priority
//  order
//  reuse
//  loop_annotations
//  annotations
// inputs
// outputs
std::string serialize(const IR& ir_) {
  auto ir = ir_;
  ir.reify_deletions();
  std::stringstream ss;
  for (const auto& v : ir.vars()) {
    ss << "v:" << ir.var(v).name() << "\n";
  }
  for (const auto& nr : ir.nodes()) {
    const auto& node = ir.node(nr);
    ss << "n:" << static_cast<int32_t>(node.op()) << ":";
    for (auto inp : node.inputs()) {
      ss << inp << ",";
    }
    ss << ":";

    for (const auto& var : node.vars()) {
      ss << var << ",";
    }
    ss << ":";

    ss << ":";  // constraints
    ss << ":";  // sym_map

    ss << ir.priority(nr) << ":";
    for (const auto& o : ir.order(nr)) {
      ss << o.first << ";" << o.second.size << ";" << o.second.tail << ",";
    }
    ss << ":";

    for (const auto& reuse : ir.not_reusable(nr)) {
      ss << reuse << ",";
    }
    ss << ":";

    for (const auto& annot : ir.loop_annotations(nr)) {
      ss << annot << ",";
    }
    ss << ":";

    ss << ir.annotation(nr) << ":";

    ss << "\n";
  }
  ss << "i:";
  for (const auto& inp : ir.inputs()) {
    ss << inp << ",";
  }
  ss << "\n";
  ss << "o:";
  for (const auto& out : ir.outputs()) {
    ss << out << ",";
  }
  ss << "\n";
  return ss.str();
}

template <typename T>
T parse_val(const std::string& s) {
  T value{};
  if (s.size()) {
    std::istringstream iss(s);
    iss >> value;
    ASSERT(!iss.fail());
  }
  return value;
};

template <typename T>
std::vector<T> parse_vec(const std::string& s) {
  std::vector<T> out;
  std::istringstream ps(s);
  for (std::string v; std::getline(ps, v, ',');) {
    out.emplace_back(parse_val<T>(v));
  }
  return out;
};

IR deserialize(const std::string& str) {
  IR ir;
  auto chunk = [](const std::string& line) -> std::vector<std::string> {
    std::vector<std::string> cs;
    std::istringstream sline(line);
    for (std::string c; std::getline(sline, c, ':');) {
      cs.emplace_back(c);
    }
    return cs;
  };

  auto parse_var = [&](const std::vector<std::string> chunks) {
    ASSERT(chunks.size() == 2) << "invalid var being parsed (found "
                               << chunks.size() << " chunks, expected 2)";
    (void)ir.create_var(chunks.at(1));
  };

  auto parse_order = [&](const std::string& s) {
    std::vector<std::pair<IR::VarRef, IR::LoopSize>> order;
    std::istringstream ss(s);
    for (std::string c; std::getline(ss, c, ',');) {
      std::vector<std::string> toks;
      std::istringstream sc(c);
      for (std::string t; std::getline(sc, t, ';');) {
        toks.emplace_back(t);
      }
      ASSERT(toks.size() == 3) << "invalid parse of order: " << s << "\n";
      auto v = parse_val<int32_t>(toks.at(0));
      auto size = parse_val<int64_t>(toks.at(1));
      auto tail = parse_val<int64_t>(toks.at(2));
      order.emplace_back(std::make_pair(v, IR::LoopSize{size, tail}));
    }
    return order;
  };

  std::vector<std::pair<IR::NodeRef, std::vector<IR::NodeRef>>> input_map;
  auto parse_node = [&](const std::vector<std::string> chunks) {
    ASSERT(chunks.size() == 11) << "invalid node being parsed (found "
                                << chunks.size() << " chunks, expected 11)";
    auto op = static_cast<Operation>(parse_val<int32_t>(chunks.at(1)));
    auto inputs = parse_vec<int32_t>(chunks.at(2));
    auto vars = parse_vec<int32_t>(chunks.at(3));
    auto node_ref = ir.create_node(op, {}, vars);
    input_map.emplace_back(node_ref, inputs);
    // TODO constraints
    // auto constraints = chunks.at(4);
    // auto sym_map = chunks.at(5);
    auto priority = parse_val<float>(chunks.at(6));
    ir.set_priority(node_ref, priority);
    auto order = parse_order(chunks.at(7));
    ir.set_order(node_ref, order);
    auto reuse = parse_vec<int32_t>(chunks.at(8));
    for (const auto& i : reuse) {
      ir.disable_reuse(node_ref, i);
    }
    auto loop_annotations = parse_vec<std::string>(chunks.at(9));
    ir.annotate_loops(node_ref, loop_annotations);
    auto annotation = chunks.at(10);
    ir.annotate(node_ref, annotation);
  };

  std::istringstream sstr(str);
  for (std::string line; std::getline(sstr, line);) {
    if (!line.size()) {
      continue;
    }
    auto chunks = chunk(line);
    if (chunks[0] == "v") {
      parse_var(chunks);
    } else if (chunks[0] == "n") {
      parse_node(chunks);
    } else if (chunks[0] == "i") {
      ir.set_inputs(parse_vec<int32_t>(chunks[1]));
    } else if (chunks[0] == "o") {
      ir.set_outputs(parse_vec<int32_t>(chunks[1]));
    }
  }
  for (const auto& p : input_map) {
    auto node_ref = p.first;
    ir.update_inputs(node_ref, p.second);
  }
  return ir;
}

}  // namespace loop_tool
