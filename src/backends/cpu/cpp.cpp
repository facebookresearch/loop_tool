/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/cpp.h"

#include <fstream>

#include "loop_tool/dynlib.h"

using namespace loop_tool;
using namespace symbolic;

CppCompiler::CppCompiler(const LoopTree &lt) : Compiler(lt) {}

bool CppCompiler::is_input_output(IR::NodeRef nr) const {
  for (auto i : lt.ir.inputs()) {
    if (nr == i) {
      return true;
    }
  }
  for (auto o : lt.ir.outputs()) {
    if (nr == o) {
      return true;
    }
  }
  return false;
};

std::string CppCompiler::gen_access_string(IR::NodeRef node_ref,
                                           LoopTree::TreeRef ref) const {
  std::stringstream ss;
  auto acc = gen_access(node_ref, ref);
  std::unordered_map<Symbol, std::string, Hash<Symbol>> sym_strings;
  auto p = lt.parent(ref);
  while (p != acc.alloc.lca) {
    const auto &l = lt.loop(p);
    auto sym = var_to_sym.at(l.var);
    std::stringstream sym_str;
    if (sym_strings.count(sym)) {
      sym_str << sym_strings.at(sym) << "+";
    }
    sym_str << "i_" << std::to_string(p);
    auto stride = inner_sizes.at(p);
    if (stride > 1) {
      sym_str << "*" << stride;
    }
    sym_strings[var_to_sym.at(l.var)] = sym_str.str();
    p = lt.parent(p);
  }
  for (const auto &p : sym_strings) {
    sym_strings[p.first] = "(" + p.second + ")";
  }

  // if we're accessing a write, we're unconstrained
  // TODO change this to allow conditional writes!
  const auto &constraints = node_ref != lt.node(ref)
                                ? get_constraints(acc)
                                : std::vector<std::pair<Expr, int64_t>>{};

  if (constraints.size()) {
    ss << "(";
  }
  for (const auto &constraint : constraints) {
    const auto &expr = constraint.first;
    const auto &str = expr.dump(false, sym_strings);
    if (&constraint != &constraints.front()) {
      ss << " && ";
    }
    ss << "((" << str << ") >= 0)";
    if (constraint.second != -1) {
      ss << " && ((" << str << ") < " << constraint.second << ")";
    }
  }
  if (constraints.size()) {
    ss << " ? ";
  }

  if (acc.alloc.size() > 1 || is_input_output(acc.alloc.node_ref)) {
    ss << "((float*)memory[" << acc.alloc.mem_idx << "])";
    ss << "[";
    for (auto i = 0; i < acc.scoped_exprs.size(); ++i) {
      if (!acc.alloc.sizes.at(i)) {
        continue;
      }
      const auto &expr = acc.scoped_exprs.at(i);
      ss << (expr * Expr(acc.alloc.size(i + 1))).dump(false, sym_strings);
      if (i != acc.scoped_exprs.size() - 1) {
        ss << "+";
      }
    }
    if (acc.scoped_exprs.size() == 0) {
      ss << "0";
    }
    ss << "]";
  } else {
    ss << "v" << acc.alloc.mem_idx;
  }

  // TODO may not be 0 when out of bounds!
  if (constraints.size()) {
    ss << " : " << 0 << ")";
  }
  return ss.str();
}

std::string CppCompiler::gen_mem_node_string(LoopTree::TreeRef ref) const {
  std::stringstream ss;
  const auto &node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);
  ASSERT(node.inputs().size() == 1);
  ss << gen_access_string(node_ref, ref);
  ss << " = ";
  ss << gen_access_string(node.inputs().at(0), ref);
  ss << ";";
  return ss.str();
}

std::string CppCompiler::gen_reset_string(LoopTree::TreeRef ref) const {
  std::stringstream ss;
  auto line_prefix = gen_indent(ref, 1);
  auto value = [&](const Node &node) -> float {
    if (node.op() == Operation::add) {
      return 0;
    } else if (node.op() == Operation::multiply) {
      return 1;
    } else if (node.op() == Operation::min) {
      return std::numeric_limits<float>::max();
    } else if (node.op() == Operation::max) {
      return -std::numeric_limits<float>::max();
    } else if (node.op() == Operation::write) {
      return 0;  // TODO fix
    } else if (node.op() == Operation::view) {
      return 0;  // TODO fix
    }
    ASSERT(0) << "cannot find default value for " << dump(node.op());
    return -1;
  };
  for (const auto &p : allocations) {
    const auto &alloc = p.second;
    if (alloc.lca == ref) {
      const auto &node = lt.ir.node(alloc.node_ref);
      bool needs_set = lt.ir.reduction_vars(alloc.node_ref).size() &&
                       node.op() != Operation::view;
      for (const auto &input : node.inputs()) {
        if (lt.ir.node(input).op() == Operation::view &&
            !lt.scheduled.count(input)) {
          needs_set = true;
        }
      }
      if (!lt.scheduled.count(alloc.node_ref)) {
        continue;
      } else if (alloc.size() == 1 && !(is_input_output(alloc.node_ref))) {
        ss << line_prefix << "float v" << alloc.mem_idx;
        if (needs_set) {
          ss << " = " << value(node);
        }
        ss << ";\n";
      } else if (needs_set) {
        set_called = true;
        ss << line_prefix << "set((float*)memory[" << alloc.mem_idx << "], ";
        ss << value(node) << ", " << alloc.size() << ");\n";
      }
    }
  }
  return ss.str();
}

std::string CppCompiler::gen_compute_node_string(LoopTree::TreeRef ref) const {
  std::stringstream ss;
  const auto &node_ref = lt.node(ref);
  const auto &node = lt.ir.node(node_ref);

  bool is_infix = [&]() {
    switch (node.op()) {
      case Operation::add:
      case Operation::multiply:
      case Operation::subtract:
      case Operation::divide:
        return true;
      default:
        return false;
    }
  }();
  bool is_binary = [&]() {
    switch (node.op()) {
      case Operation::add:
      case Operation::multiply:
      case Operation::subtract:
      case Operation::divide:
      case Operation::min:
      case Operation::max:
        return true;
      default:
        return false;
    }
  }();
  auto op = [&]() {
    switch (node.op()) {
      case Operation::add:
        return "+";
      case Operation::multiply:
        return "*";
      case Operation::subtract:
        return "-";
      case Operation::divide:
        return "/";
      case Operation::max:
        return "max";
      case Operation::min:
        return "min";
      case Operation::log:
        return "log";
      case Operation::exp:
        return "exp";
      case Operation::sqrt:
        return "sqrt";
      case Operation::negate:
        return "-";
      case Operation::abs:
        return "abs";
      case Operation::reciprocal:
        return "1 / ";
      default:
        ASSERT(0) << "can't emit code for " << dump(node.op());
        return "";
    }
  }();

  ss << gen_access_string(node_ref, ref);
  ss << " = ";

  bool is_reduction = lt.ir.reduction_vars(node_ref).size();
  std::vector<std::string> access_strings;
  if (is_reduction) {
    access_strings.emplace_back(gen_access_string(node_ref, ref));
  }
  for (const auto &inp : node.inputs()) {
    access_strings.emplace_back(gen_access_string(inp, ref));
  }

  if (is_infix) {
    for (const auto &access_string : access_strings) {
      ss << access_string;
      if (&access_string != &access_strings.back()) {
        ss << " " << op << " ";
      }
    }
  } else if (is_binary) {
    std::function<void(int)> nest;
    nest = [&](int i) {
      if (i == access_strings.size() - 1) {
        ss << access_strings.at(i);
        return;
      }
      ss << op << "(" << access_strings.at(i) << ", ";
      nest(i + 1);
      ss << ")";
    };
    nest(0);
  } else {
    ASSERT(access_strings.size() == 1);
    ss << op << "(" << access_strings.at(0) << ")";
  }
  ss << ";";
  return ss.str();
}

std::string CppCompiler::gen_node_string(LoopTree::TreeRef ref) const {
  std::stringstream ss;
  auto line_prefix = gen_indent(ref);
  const auto &node = lt.ir.node(lt.node(ref));

  if (lt.children(lt.parent(ref)).at(0) == ref) {
    ss << gen_reset_string(lt.parent(ref));
  }
  ss << line_prefix;
  switch (node.op()) {
    case Operation::write:
    case Operation::view:
      ss << gen_mem_node_string(ref);
      break;
    case Operation::read:
      break;
    default:
      ss << gen_compute_node_string(ref);
  }
  ss << " // %" << lt.node(ref) << " (" << dump(node.op()) << ")\n";
  return ss.str();
}

std::string CppCompiler::gen_loop_string(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  std::stringstream ss;
  auto line_prefix = gen_indent(ref);

  const auto &loop = lt.loop(ref);
  std::string iter_var = "i_" + std::to_string(ref);

  ASSERT(loop.size > -1);
  ASSERT(loop.tail > -1);
  int size = loop.size;
  int tail = loop.tail;

  // if there's an override, take it
  if (overrides.count(loop.var)) {
    auto override_size = overrides.at(loop.var);
    auto inner_size = inner_sizes.at(ref);
    size = override_size / inner_size;
    tail = override_size % inner_size;
    overrides.erase(loop.var);
  }

  std::vector<std::string> body_children;
  std::vector<std::string> tail_children;
  for (auto c : lt.children(ref)) {
    body_children.emplace_back(gen_string_impl(c, overrides));
  }
  if (tail > 0) {
    // find first loop of same var, and override
    overrides[loop.var] = tail;
    for (const auto &cref : lt.children(ref)) {
      tail_children.emplace_back(gen_string_impl(cref, overrides));
    }
  }

  if (lt.children(lt.parent(ref)).at(0) == ref) {
    ss << gen_reset_string(lt.parent(ref));
  }
  ss << line_prefix << "for (int64_t " << iter_var << " = 0L; ";
  ss << iter_var << " < " << size << "L; ++" << iter_var << ") { // "
     << lt.ir.var(loop.var).name() << "\n";
  for (auto c : body_children) {
    ss << c;
  }
  ss << line_prefix << "}\n";
  if (tail > 0) {
    ss << line_prefix << "{ // " << lt.ir.var(loop.var).name() << " tail\n";
    ss << gen_indent(ref, 1) << "int64_t " << iter_var << " = " << loop.size
       << "L;\n";
    for (auto c : tail_children) {
      ss << c;
    }
    ss << line_prefix << "}\n";
  }
  return ss.str();
}

std::string CppCompiler::gen_string_impl(
    LoopTree::TreeRef ref,
    std::unordered_map<IR::VarRef, int> overrides) const {
  if (ref == -1) {
    // generate the body first to minimize header code
    std::stringstream body;
    for (auto c : lt.roots) {
      body << gen_string_impl(c);
    }
    std::stringstream ss;
    bool define_max = false;
    bool define_min = false;
    for (auto n : lt.ir.nodes()) {
      if (lt.ir.node(n).op() == Operation::max) {
        define_max = true;
      }
      if (lt.ir.node(n).op() == Operation::min) {
        define_min = true;
      }
    }

    ss << R"""(#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

)""";
    if (define_max) {
      ss << R"""(
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
)""";
    }
    if (define_min) {
      ss << R"""(
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
)""";
    }

    if (set_called) {
      ss << R"""(
static inline void set(float* mem, float val, int64_t length) {
  for (int64_t i = 0; i < length; ++i) {
    mem[i] = val;
  }
}
)""";
    }

    ss << "\n";
    const auto &sizes = memory_sizes();
    auto i = 0;
    auto num_inputs = lt.ir.inputs().size();
    auto num_outputs = lt.ir.outputs().size();
    auto dump = [&](int idx, int64_t s, bool force_nonnull = false) {
      ss << idx << ":";
      if (s <= 1 && !force_nonnull) {
        ss << "nullptr";
      } else {
        ss << "float[" << s << "]";
      }
      ss << ", ";
    };
    ss << "// memory: {\n";
    ss << "//   ";
    for (; i < num_inputs; ++i) {
      dump(i, sizes.at(i), true);
    }
    ss << "// inputs\n";
    ss << "//   ";
    for (; i < num_inputs + num_outputs; ++i) {
      dump(i, sizes.at(i), true);
    }
    ss << "// outputs\n";
    ss << "//   ";
    for (; i < sizes.size(); ++i) {
      dump(i, sizes.at(i));
    }
    ss << "// scratch\n";
    ss << "// }\n";
    ss << "void fn_" << count << "(void** memory) {\n";
    ss << body.str();
    ss << "}\n";
    return ss.str();
  }
  if (lt.kind(ref) == LoopTree::NODE) {
    return gen_node_string(ref);
  }
  return gen_loop_string(ref, overrides);
}

struct CppCompiled : public Compiled {
  std::vector<int64_t> intermediates;
  InnerFnType fn;
  std::string code;
  mutable std::vector<void *> mem;
  mutable std::vector<int64_t> mem_sizes;
  std::shared_ptr<loop_tool::DynamicLibrary> dll;

  CppCompiled(const LoopTree &lt) {
    auto compiler = CppCompiler(lt);
    code = compiler.gen_string();
    try {
      std::stringstream fn_name;
      fn_name << "fn_" << compiler.count;
      std::string source_name = "/tmp/" + fn_name.str() + ".c";
      std::string lib_name = "/tmp/" + fn_name.str() + ".so";
      std::ofstream(source_name, std::ios::trunc) << code;
      std::string compile_call =
          "cc -Wall -Wno-unused-function -Wno-unused-variable -Werror -O3 "
          "-std=c99 -fpic -shared -o " +
          lib_name + " " + source_name;
      ASSERT(!std::system(compile_call.c_str()));
      dll = std::make_shared<loop_tool::DynamicLibrary>(lib_name.c_str());
      auto fn_impl = dll->sym<void (*)(void **)>(fn_name.str().c_str());
      fn = [=](const std::vector<void *> &memory, int indices[MAX_DEPTH]) {
        fn_impl(const_cast<void **>(memory.data()));
      };
      std::remove(source_name.c_str());
    } catch (const std::exception &e) {
      std::cerr << "Error compiling, falling back to interpreted...\n";
      fn = compiler.gen_exec();
    }

    mem_sizes = compiler.memory_sizes();
    mem = allocate(mem_sizes);
  }

  ~CppCompiled() {
    for (auto i = 0; i < mem_sizes.size(); ++i) {
      if (mem_sizes[i] > 0) {
        free(mem[i]);
      }
    }
  }

  void run(const std::vector<void *> &memory, bool sync) const override {
    int indices[MAX_DEPTH] = {0};
    for (auto i = 0; i < memory.size(); ++i) {
      mem[i] = memory[i];
    }
    fn(mem, indices);
  }

  std::string dump() const override { return code; }
};

std::unique_ptr<Compiled> CppBackend::compile_impl(const LoopTree &lt) {
  return std::make_unique<CppCompiled>(lt);
}

int CppBackend::hardware_requirement() const {
  // CPU is the only guaranteed hardware, always id = 0
  return 1 << 0;
}

static RegisterBackend cpu_backend_reg_(std::make_shared<CppBackend>());
