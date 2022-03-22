#include "loop_tool/serialization.h"

namespace loop_tool {

using namespace loop_tool::symbolic;

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
std::vector<T> parse_vec(const std::string& s, char delim = ',') {
  std::vector<T> out;
  std::istringstream ps(s);
  for (std::string v; std::getline(ps, v, delim);) {
    out.emplace_back(parse_val<T>(v));
  }
  return out;
};

template <typename T, typename U>
std::unordered_map<T, U> parse_map(const std::string& s) {
  std::unordered_map<T, U> out;
  std::istringstream ps(s);
  for (std::string v; std::getline(ps, v, ',');) {
    auto pos = v.find('=');
    auto lhs = v.substr(0, pos);
    auto rhs = v.substr(pos + 1);
    out.emplace(parse_val<T>(lhs), parse_val<U>(rhs));
  }
  return out;
};

Expr deserialize_expr(const std::string& str,
                      std::unordered_map<int32_t, Symbol>& idx_to_sym,
                      int32_t& idx) {
  ASSERT(str.size());
  std::istringstream sstr(str);
  std::unordered_map<int32_t, Expr> idx_map;
  for (std::string line; std::getline(sstr, line, ';');) {
    std::vector<std::string> toks;
    std::istringstream sline(line);
    for (std::string tok; std::getline(sline, tok, '|');) {
      toks.emplace_back(tok);
    }
    ASSERT(toks.size() == 3);
    auto op = parse_val<int32_t>(toks.at(0));
    auto type = toks.at(1);
    if (type == "s") {
      auto sym = Symbol(toks.at(2));
      idx_map.emplace(idx, Expr(sym));
      idx_to_sym.emplace(idx, sym);
    } else if (type == "r") {
      auto sym_idx = parse_val<int32_t>(toks.at(2));
      auto sym = idx_to_sym.at(sym_idx);
      idx_map.emplace(idx, Expr(sym));
      idx_to_sym.emplace(idx, sym);
    } else if (type == "v") {
      idx_map.emplace(idx, Expr(parse_val<int64_t>(toks.at(2))));
    } else if (type == "f") {
      auto arg_idxs = parse_vec<int32_t>(toks.at(2));
      std::vector<Expr> args;
      for (const auto& idx : arg_idxs) {
        if (idx_to_sym.count(idx)) {
          args.emplace_back(Expr(idx_to_sym.at(idx)));
        } else {
          ASSERT(idx_map.count(idx))
              << "can't find idx " << idx << " in map " << idx_map.size();
          args.emplace_back(idx_map.at(idx));
        }
      }
      idx_map.emplace(idx, Expr(static_cast<Op>(op), args));
    } else {
      ASSERT(0) << "invalid type " << type;
    }
    idx++;
  }
  return idx_map.at(idx - 1);
}

std::string serialize(const Expr& expr,
                      std::unordered_map<int32_t, int32_t>& sym_id_to_idx,
                      int32_t& idx) {
  std::vector<std::string> serializations;

  std::function<int32_t(const Expr&)> emit;
  emit = [&](const Expr& e) -> int32_t {
    std::stringstream ss;
    if (e.type() == Expr::Type::function) {
      ss << static_cast<int32_t>(e.op()) << "|f|";
      for (const auto& arg : e.args()) {
        auto arg_idx = emit(arg);
        ss << arg_idx << ",";
      }
      ss << "|";
      serializations.emplace_back(ss.str());
      return idx++;
    } else if (e.type() == Expr::Type::value) {
      ss << static_cast<int32_t>(e.op()) << "|v|" << e.value() << "|";
      serializations.emplace_back(ss.str());
      return idx++;
    }
    ASSERT(e.type() == Expr::Type::symbol);
    const auto& sym = e.symbol();
    if (!sym_id_to_idx.count(sym.id())) {
      sym_id_to_idx[sym.id()] = idx++;
      ss << static_cast<int32_t>(e.op()) << "|s|" << sym.name() << "|";
      serializations.emplace_back(ss.str());
    } else {
      ss << static_cast<int32_t>(e.op()) << "|r|" << sym_id_to_idx.at(sym.id())
         << "|";
      serializations.emplace_back(ss.str());
      idx++;
    }
    return sym_id_to_idx.at(sym.id());
  };
  emit(expr);
  std::stringstream ss;
  for (const auto& s : serializations) {
    ss << s << ";";
  }
  return ss.str();
}

std::string serialize(const Constraint& c,
                      std::unordered_map<int32_t, int32_t>& sym_id_to_idx,
                      int32_t& idx) {
  auto lhs = serialize(c.first, sym_id_to_idx, idx);
  ASSERT(lhs.size()) << "couldn't serialize " << c.first.dump();
  auto rhs = serialize(c.second, sym_id_to_idx, idx);
  ASSERT(rhs.size()) << "couldn't serialize " << c.second.dump();
  return lhs + "=" + rhs;
}

Constraint deserialize_constraint(
    const std::string& s, std::unordered_map<int32_t, Symbol>& idx_to_sym,
    int32_t& idx) {
  std::vector<std::string> toks;
  auto pos = s.find('=');
  auto lhs = s.substr(0, pos);
  auto rhs = s.substr(pos + 1);
  auto lhse = deserialize_expr(lhs, idx_to_sym, idx);
  auto rhse = deserialize_expr(rhs, idx_to_sym, idx);
  return std::make_pair(lhse, rhse);
}

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

    std::unordered_map<int32_t, int32_t> id_to_idx;
    int32_t idx = 0;
    for (const auto& c : node.constraints()) {
      ss << serialize(c, id_to_idx, idx) << "/";
    }
    ss << ":";
    for (const auto& p : node.sym_to_var()) {
      ss << id_to_idx.at(p.first) << "=" << p.second << ",";
    }
    ss << ":";

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
    // TODO constraints
    auto constraint_strs = parse_vec<std::string>(chunks.at(4), '/');
    std::vector<Constraint> constraints;
    std::unordered_map<int32_t, Symbol> idx_to_sym;
    int32_t idx = 0;
    for (const auto& cs : constraint_strs) {
      constraints.emplace_back(deserialize_constraint(cs, idx_to_sym, idx));
    }
    auto idx_to_var = parse_map<int32_t, int32_t>(chunks.at(5));
    std::unordered_map<int32_t, IR::VarRef> sym_map;
    for (const auto& p : idx_to_var) {
      sym_map[idx_to_sym.at(p.first).id()] = p.second;
    }
    auto node_ref = ir.create_node(op, {}, vars, constraints, sym_map);
    input_map.emplace_back(node_ref, inputs);

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

// op:t(.*)
// 0:sym:name;
// 0:val:number;
// #:fn:0,1,;

}  // namespace loop_tool
