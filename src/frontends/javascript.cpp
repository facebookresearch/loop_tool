/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "emscripten/bind.h"
#include "emscripten/val.h"
#include "loop_tool/loop_tool.h"
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

lazy::Tensor to_impl(const lazy::Tensor &t, std::vector<lazy::Symbol> symbols) {
  return t.to(symbols);
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

emscripten::val wasm(const lazy::Tensor &t) {
  auto wc = loop_tool::WebAssemblyCompiler(t.loop_tree());
  auto bytes = wc.emit();
  emscripten::val view{
      emscripten::typed_memory_view(bytes.size(), bytes.data())};
  auto result = emscripten::val::global("Uint8Array").new_(bytes.size());
  // copy data from generated output to return object
  result.call<void>("set", view);
  return result;
}

EMSCRIPTEN_BINDINGS(loop_tool) {
  js::class_<lazy::Symbol>("Symbol")
      .constructor<std::string>()
      .function("name", &lazy::Symbol::name)
      .function("id", &lazy::Symbol::id);
  js::class_<lazy::Tensor>("Tensor")
      .constructor(&tensor_constructor, emscripten::allow_raw_pointers())
      .function("to", &to_impl)
      .function("as", &as_impl)
      .function("shape", &sizes_impl)
      .function("symbolic_shape", &lazy::Tensor::shape)
      .function("add", &lazy::Tensor::operator+)
      .function(
          "sub",
          emscripten::select_overload<lazy::Tensor(const lazy::Tensor &) const>(
              &lazy::Tensor::operator-))
      .function("mul", &lazy::Tensor::operator*)
      .function("div", &lazy::Tensor::operator/)
      .function("max", &lazy::Tensor::max)
      .function("min", &lazy::Tensor::min)
      .function("sum", &sum_impl)
      .function("neg", emscripten::select_overload<lazy::Tensor() const>(
                           &lazy::Tensor::operator-))
      .function("abs", &lazy::Tensor::abs)
      .function("graphviz", &graphviz)
      .function("numel", &lazy::Tensor::numel)
      .function("code", &lazy::Tensor::code)
      .function("hash", &lazy::Tensor::hash)
      .function("wasm", &wasm);
  emscripten::function("getExceptionMessage", &getExceptionMessage);
}
