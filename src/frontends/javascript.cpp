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

lazy::Tensor *tensor_constructor(const std::vector<int32_t> &sv) {
  std::vector<int64_t> inp;
  for (auto s : sv) {
    inp.emplace_back(s);
  }
  return new lazy::Tensor(inp);
}

std::vector<int32_t> sizes_impl(const lazy::Tensor &t) {
  std::vector<int32_t> out;
  for (const auto &s : t.sizes()) {
    out.emplace_back(s);
  }
  return out;
}

std::string getExceptionMessage(int ptr) {
  return std::string(reinterpret_cast<std::exception *>(ptr)->what());
}

std::string graphviz(const lazy::Tensor &t) { return dot(t.ir()); }

emscripten::val wasm(const LoopTree &loop_tree) {
  auto wc = loop_tool::WebAssemblyCompiler(loop_tree);
  auto bytes = wc.emit();
  emscripten::val view{
      emscripten::typed_memory_view(bytes.size(), bytes.data())};
  auto result = emscripten::val::global("Uint8Array").new_(bytes.size());
  // copy data from generated output to return object
  result.call<void>("set", view);
  return result;
}

lazy::Expr expr_from_sym(lazy::Symbol sym) { return lazy::Expr(sym); }

std::string dump_expr(lazy::Expr &e) { return e.dump(); }

std::string dump_loop_tree(const LoopTree &lt) { return lt.dump(); }

std::vector<LoopTree::TreeRef> walk_loop_tree(const LoopTree &lt) {
  std::vector<LoopTree::TreeRef> out;
  auto fn = [&](LoopTree::TreeRef ref, int depth) { out.emplace_back(ref); };
  lt.walk(fn);
  return out;
}

bool is_loop(const LoopTree &lt, LoopTree::TreeRef ref) {
  return lt.kind(ref) == LoopTree::LOOP;
}

IR::VarRef loop_var(const LoopTree::Loop &loop) { return loop.var; }

int32_t loop_size(const LoopTree::Loop &loop) { return loop.size; }

int32_t loop_tail(const LoopTree::Loop &loop) { return loop.tail; }

EMSCRIPTEN_BINDINGS(loop_tool) {
  js::class_<lazy::Expr>("Expr")
      .constructor<int>()
      .function("dump", &dump_expr)
      .function("add", &lazy::Expr::operator+)
      .function("mul", &lazy::Expr::operator*)
      .function("hash", &lazy::Symbol::hash);
  js::class_<lazy::Symbol>("Symbol")
      .constructor<std::string>()
      .function("name", &lazy::Symbol::name)
      .function("expr", &expr_from_sym)
      .function("id", &lazy::Symbol::id);
  js::class_<LoopTree::Loop>("Loop")
      .function("var", &loop_var)
      .function("size", &loop_size)
      .function("tail", &loop_tail);
  js::class_<LoopTree>("LoopTree")
      .function("dump", &dump_loop_tree)
      .function("wasm", &wasm)
      .function("walk", &walk_loop_tree)
      .function("depth", &LoopTree::depth)
      .function("children", &LoopTree::children)
      .function("is_loop", &is_loop)
      .function("loop", &LoopTree::loop)
      .function("node", &LoopTree::node)
      .function("annotation", &LoopTree::annotation)
      .function("annotate", &LoopTree::annotate);
  js::class_<lazy::Tensor>("Tensor")
      .constructor(&tensor_constructor, js::allow_raw_pointers())
      .function("to", &to_impl)
      .function("as", &as_impl)
      .function("shape", &sizes_impl)
      .function("symbolic_shape", &lazy::Tensor::shape)
      .function("add", &lazy::Tensor::operator+)
      .function("sub",
                js::select_overload<lazy::Tensor(const lazy::Tensor &) const>(
                    &lazy::Tensor::operator-))
      .function("mul", &lazy::Tensor::operator*)
      .function("div", &lazy::Tensor::operator/)
      .function("max", &lazy::Tensor::max)
      .function("min", &lazy::Tensor::min)
      .function("sum", &sum_impl)
      .function("neg", js::select_overload<lazy::Tensor() const>(
                           &lazy::Tensor::operator-))
      .function("abs", &lazy::Tensor::abs)
      .function("graphviz", &graphviz)
      .function("loop_tree", &lazy::Tensor::loop_tree)
      .function("numel", &lazy::Tensor::numel)
      .function("code", &lazy::Tensor::code)
      .function("hash", &lazy::Tensor::hash);
  js::function("split", split);
  js::function("swap", swap);
  js::function("disable_reuse", disable_reuse);
  js::function("enable_reuse", enable_reuse);
  js::function("getExceptionMessage", &getExceptionMessage);
}
