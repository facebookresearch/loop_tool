/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "emscripten/bind.h"
#include "loop_tool/compile.h"
#include "loop_tool/ir.h"

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

} // namespace internal
} // namespace emscripten

std::string dump(const LoopTree &lt) { return lt.dump(); }

EMSCRIPTEN_BINDINGS(loop_tool) {
  // js::register_vector<int>("VectorInt");
  js::class_<IR>("IR")
      .constructor<>()
      .function("create_var", &IR::create_var)
      .function("create_node", &IR::create_node)
      .function("set_inputs", &IR::set_inputs)
      .function("set_outputs", &IR::set_outputs)
      .function("set_priority", &IR::set_priority);
  js::class_<LoopTree>("LoopTree")
      .constructor<const IR &>()
      .function("dump", &dump);
}
