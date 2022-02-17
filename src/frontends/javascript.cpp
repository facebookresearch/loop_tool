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

lazy::Tensor *makeTensor(const std::vector<int32_t> &sv) {
  std::vector<int64_t> inp;
  for (auto s : sv) {
    inp.emplace_back(s);
  }
  return new lazy::Tensor(inp);
}

std::string getExceptionMessage(int ptr) {
  return std::string(reinterpret_cast<std::exception *>(ptr)->what());
}

std::string graphviz(const lazy::Tensor &t) { return dot(t.ir()); }

emscripten::val wasm(const lazy::Tensor &t) {
  auto wc = loop_tool::WebAssemblyCompiler(t.loop_tree());
  auto bytes = wc.emit();
  return emscripten::val(
     emscripten::typed_memory_view(bytes.size(),
                                   bytes.data()));
}

EMSCRIPTEN_BINDINGS(loop_tool) {
  js::class_<lazy::Symbol>("Symbol").constructor<std::string>();
  js::class_<lazy::Tensor>("Tensor")
      .constructor(&makeTensor, emscripten::allow_raw_pointers())
      .function("to", &to_impl)
      .function("as", &as_impl)
      .function("mul", &lazy::Tensor::operator*)
      .function("add", &lazy::Tensor::operator+)
      .function("graphviz", &graphviz)
      .function("code", &lazy::Tensor::code)
      .function("wasm", &wasm);
  emscripten::function("getExceptionMessage", &getExceptionMessage);
}
