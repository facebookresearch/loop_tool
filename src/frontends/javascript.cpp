/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "emscripten/bind.h"
#include "emscripten/val.h"
#include "loop_tool/loop_tool.h"
#include "loop_tool/mutate.h"
#include "loop_tool/wasm.h"

using namespace loop_tool;

namespace js = emscripten;

namespace emscripten {
namespace internal {

template <typename T, typename Allocator>
struct BindingType<std::vector<T, Allocator>> {
  using ValBinding = BindingType<val>;
  using WireType = ValBinding::WireType;

  static WireType toWireType(const std::vector<T, Allocator> &vec) {
    return ValBinding::toWireType(val::array(vec));
  }

  static std::vector<T, Allocator> fromWireType(WireType value) {
    return vecFromJSArray<T>(ValBinding::fromWireType(value));
  }
};

template <typename T>
struct TypeID<
    T,
    typename std::enable_if_t<std::is_same<
        typename Canonicalized<T>::type,
        std::vector<typename Canonicalized<T>::type::value_type,
                    typename Canonicalized<T>::type::allocator_type>>::value>> {
  static constexpr TYPEID get() { return TypeID<val>::get(); }
};

}  // namespace internal
}  // namespace emscripten

std::string dump(const LoopTree &lt) { return lt.dump(); }

lazy::Tensor to_impl(const lazy::Tensor &t, std::vector<lazy::Symbol> symbols,
                     std::vector<std::vector<lazy::Expr>> constraints) {
  std::vector<lazy::Constraint> real_constraints;
  for (const auto &v : constraints) {
    ASSERT(v.size() == 2) << "invalid constraint parameter to Tensor.to(), "
                             "expected list of size 2";
    real_constraints.emplace_back(std::make_pair(v.at(0), v.at(1)));
  }
  return t.to(symbols, real_constraints);
}

lazy::Tensor as_impl(const lazy::Tensor &t, std::vector<lazy::Symbol> symbols) {
  return t.as(symbols);
}

lazy::Tensor sum_impl(const lazy::Tensor &t,
                      const std::vector<lazy::Symbol> &symbols) {
  return t.sum(symbols);
}

lazy::Tensor max_impl(const lazy::Tensor &t,
                      const std::vector<lazy::Symbol> &symbols) {
  return t.max(symbols);
}

lazy::Tensor min_impl(const lazy::Tensor &t,
                      const std::vector<lazy::Symbol> &symbols) {
  return t.min(symbols);
}

lazy::Tensor prod_impl(const lazy::Tensor &t,
                       const std::vector<lazy::Symbol> &symbols) {
  return t.prod(symbols);
}

lazy::Tensor flatten_impl(const lazy::Tensor &t,
                          const std::vector<lazy::Symbol> &symbols,
                          const lazy::Symbol &sym) {
  return t.flatten(symbols, sym);
}

lazy::Tensor sqrt_impl(const lazy::Tensor &t) { return t.sqrt(); }

lazy::Tensor pad_impl(const lazy::Tensor &t, lazy::Symbol sym, int32_t pre,
                      int32_t post) {
  return t.pad(sym, pre, post);
}

lazy::Tensor tensor_constructor(const std::vector<int32_t> &sv) {
  std::vector<int64_t> inp;
  for (auto s : sv) {
    inp.emplace_back(s);
  }
  return lazy::Tensor(inp);
}

std::vector<int32_t> sizes_impl(const lazy::Tensor &t) {
  std::vector<int32_t> out;
  for (const auto &s : t.sizes()) {
    out.emplace_back(s);
  }
  return out;
}

std::vector<js::val> shape_impl(const lazy::Tensor &t) {
  std::vector<js::val> out;
  for (const auto &s : t.shape()) {
    out.emplace_back(s);
  }
  return out;
}

std::string getExceptionMessage(int ptr) {
  return std::string(reinterpret_cast<std::exception *>(ptr)->what());
}

std::string graphviz(const lazy::Tensor &t) { return dot(t.ir()); }

js::val wasm(const LoopTree &loop_tree) {
  auto wc = loop_tool::WebAssemblyCompiler(loop_tree);
  auto bytes = wc.emit();
  js::val view{js::typed_memory_view(bytes.size(), bytes.data())};
  auto result = js::val::global("Uint8Array").new_(bytes.size());
  // copy data from generated output to return object
  result.call<void>("set", view);
  return result;
}

std::string c_code(const LoopTree &loop_tree) {
  auto cc = Compiler(loop_tree);
  return cc.gen_string();
}

int32_t node_size(const LoopTree &lt, IR::NodeRef node_ref) {
  auto wc = loop_tool::WebAssemblyCompiler(lt);
  return wc.allocations.at(node_ref).size();
}

std::string node_attributes(const LoopTree &lt, IR::NodeRef node_ref) {
  std::stringstream ss;
  auto wc = loop_tool::WebAssemblyCompiler(lt);
  if (wc.is_local(node_ref)) {
    ss << "local";
  }
  if (wc.is_on_stack(node_ref)) {
    ss << "stack";
  }
  if (wc.is_vector_stored(node_ref)) {
    auto var_name = lt.ir.var(wc.vector_storage_dim(node_ref)).name();
    ss << ", vectorized";
    auto var_sym_name = var_name.substr(0, var_name.find("_"));
    ss << " over " << var_sym_name;
    if (wc.is_broadcast(node_ref)) {
      ss << ", broadcast";
    }
  }
  return ss.str();
}

lazy::Expr expr_from_sym(lazy::Symbol sym) { return lazy::Expr(sym); }

lazy::Expr expr_constructor(int v) { return lazy::Expr((int64_t)v); }

lazy::Expr to_size_expr(lazy::Symbol sym) { return lazy::Expr::size(sym); }

std::string dump_expr(lazy::Expr &e) { return e.dump(); }

std::string dump_loop_tree(const LoopTree &lt) { return lt.dump(); }
std::string serialize_loop_tree(const LoopTree &lt) { return serialize(lt.ir); }
LoopTree deserialize_loop_tree(const std::string &s) {
  return LoopTree(deserialize(s));
}

std::vector<LoopTree::TreeRef> walk_loop_tree(const LoopTree &lt) {
  std::vector<LoopTree::TreeRef> out;
  auto fn = [&](LoopTree::TreeRef ref, int depth) { out.emplace_back(ref); };
  lt.walk(fn);
  return out;
}

float get_flops(const LoopTree &lt) { return (float)FLOPs(lt); }

LoopTree split_impl(const LoopTree &lt, LoopTree::TreeRef ref, int32_t size) {
  return split(lt, ref, size);
}

bool is_loop(const LoopTree &lt, LoopTree::TreeRef ref) {
  return lt.kind(ref) == LoopTree::LOOP;
}

std::string var_name(const LoopTree &lt, IR::VarRef v) {
  return lt.ir.var(v).name();
}

std::string dump_node(const LoopTree &lt, IR::NodeRef n) {
  return lt.ir.dump(n);
}

std::string node_type(const LoopTree &lt, IR::NodeRef n) {
  return dump(lt.ir.node(n).op());
}

std::vector<IR::VarRef> node_vars(const LoopTree &lt, IR::NodeRef n) {
  return lt.ir.node(n).vars();
}

std::vector<IR::NodeRef> node_outputs(const LoopTree &lt, IR::NodeRef n) {
  return lt.ir.node(n).outputs();
}

std::vector<IR::NodeRef> node_inputs(const LoopTree &lt, IR::NodeRef n) {
  return lt.ir.node(n).inputs();
}

IR::VarRef loop_var(const LoopTree::Loop &loop) { return loop.var; }

int32_t loop_size(const LoopTree::Loop &loop) { return loop.size; }

int32_t loop_tail(const LoopTree::Loop &loop) { return loop.tail; }

EMSCRIPTEN_BINDINGS(loop_tool) {
  js::class_<lazy::Expr>("Expr")
      .constructor(&expr_constructor)
      .function("dump", &dump_expr)
      .function("add", &lazy::Expr::operator+)
      .function("sub",
                js::select_overload<lazy::Expr(const lazy::Expr &) const>(
                    &lazy::Expr::operator-))
      .function("mul", &lazy::Expr::operator*)
      .function("div", &lazy::Expr::operator/)
      .function("hash", &lazy::Symbol::hash);
  js::class_<lazy::Symbol>("Symbol")
      .constructor<std::string>()
      .function("name", &lazy::Symbol::name)
      .function("expr", &expr_from_sym)
      .function("id", &lazy::Symbol::id);
  js::class_<LoopTree::Loop>("Loop")
      .function("v", &loop_var)
      .function("size", &loop_size)
      .function("tail", &loop_tail);
  js::class_<LoopTree>("LoopTree")
      .function("dump", &dump_loop_tree)
      .function("serialize", &serialize_loop_tree)
      .function("wasm", &wasm)
      .function("code", &c_code)
      .function("walk", &walk_loop_tree)
      .function("flops", &get_flops)
      .function("next_ref", &next_ref)
      .function("prev_ref", &previous_ref)
      .function("depth", &LoopTree::depth)
      .function("children", &LoopTree::children)
      .function("parent", &LoopTree::parent)
      .function("is_loop", &is_loop)
      .function("loop", &LoopTree::loop)
      .function("node", &LoopTree::node)
      .function("node_outputs", &node_outputs)
      .function("node_inputs", &node_inputs)
      .function("node_vars", &node_vars)
      .function("node_type", &node_type)
      .function("node_size", &node_size)
      .function("node_attributes", &node_attributes)
      .function("node_s", &dump_node)
      .function("var_name", &var_name)
      .function("annotation", &LoopTree::annotation)
      .function("annotate", &annotate)
      .function("split", split_impl)
      .function("merge", merge)
      .function("try_swap", try_swap)
      .function("swap_loops", swap_loops)
      .function("swap_nodes", swap_nodes)
      .function(
          "swap_vars",
          js::select_overload<LoopTree(const LoopTree &, IR::NodeRef,
                                       IR::VarRef, IR::VarRef)>(swap_vars))
      .function("copy_input", copy_input)
      .function("delete_copy", delete_copy)
      .function("add_loop", add_loop)
      .function("remove_loop", remove_loop)
      .function("disable_reuse", disable_reuse)
      .function("enable_reuse", enable_reuse)
      .function("decrease_reuse", decrease_reuse)
      .function("increase_reuse", increase_reuse)
      .function("maximize_reuse", maximize_reuse)
      .function("unroll_inner_loops", unroll_inner_loops)
      .function("map_ref", map_ref);
  js::class_<lazy::Tensor>("Tensor")
      .constructor(&tensor_constructor)
      .function("to", &to_impl)
      .function(
          "transpose",
          js::select_overload<lazy::Tensor(std::vector<lazy::Symbol>) const>(
              &lazy::Tensor::transpose))
      .function("as", &as_impl)
      .function("shape", &sizes_impl)
      .function("symbolic_shape", &shape_impl)
      .function("add", &lazy::Tensor::operator+)
      .function("sub",
                js::select_overload<lazy::Tensor(const lazy::Tensor &) const>(
                    &lazy::Tensor::operator-))
      .function("mul", &lazy::Tensor::operator*)
      .function("div", &lazy::Tensor::operator/)
      .function("max",
                js::select_overload<lazy::Tensor(const lazy::Tensor &) const>(
                    &lazy::Tensor::max))
      .function("min",
                js::select_overload<lazy::Tensor(const lazy::Tensor &) const>(
                    &lazy::Tensor::min))
      .function("sum", &sum_impl)
      .function("max_reduce", &max_impl)
      .function("min_reduce", &min_impl)
      .function("prod", &prod_impl)
      .function("flatten", &flatten_impl)
      .function("neg", js::select_overload<lazy::Tensor() const>(
                           &lazy::Tensor::operator-))
      .function("sqrt", &sqrt_impl)
      .function("abs", &lazy::Tensor::abs)
      .function("pad", &pad_impl)
      .function("graphviz", &graphviz)
      .function("loop_tree", &lazy::Tensor::loop_tree)
      .function("numel", &lazy::Tensor::numel)
      .function("hash", &lazy::Tensor::hash);
  js::function("size", &to_size_expr);
  js::function("deserialize", &deserialize_loop_tree);
  js::function("getExceptionMessage", &getExceptionMessage);
}
