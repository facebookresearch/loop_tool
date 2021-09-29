/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>
#include <sstream>

#include "loop_tool/backend.h"
#include "loop_tool/compile.h"
#include "loop_tool/error.h"
#include "loop_tool/ir.h"
#include "loop_tool/lazy.h"
#include "loop_tool/tensor.h"

#ifdef ENABLE_CUDA
#include <cuda.h>

#include "cuda_backend.h"
#endif

using namespace loop_tool;
namespace py = pybind11;

static int default_hardware_id = 0;  // CPU
static bool cuda_available = false;
PYBIND11_MODULE(loop_tool_py, m) {
  m.def("backends", []() {
    std::vector<std::string> backends = {"cpu"};
    for (const auto &hw : getHardware()) {
      if (hw->name() == "cuda") {
        backends.emplace_back("cuda");
        cuda_available = true;
      }
    }
    return backends;
  });
  m.def("set_default_hardware", [](std::string hardware) {
    bool set = false;
    for (auto &hw : getHardware()) {
      if (hw->name() == hardware) {
        default_hardware_id = hw->id();
        set = true;
      }
    }
    ASSERT(set) << "cannot find hardware: " << hardware;
  });
  py::enum_<Operation>(m, "Operation")
#define X(op) .value(#op, Operation::op)
      OPS(X)
#undef X
          .export_values();
  py::class_<IR>(m, "IR")
      .def(py::init<>())
      .def("create_var", &IR::create_var)
      .def("create_node", &IR::create_node, py::arg("op"), py::arg("inputs"),
           py::arg("vars"),
           py::arg("constraints") = std::vector<symbolic::Constraint>{})
      .def("set_inputs", &IR::set_inputs)
      .def("set_outputs", &IR::set_outputs)
      .def("set_priority", &IR::set_priority)
      .def("set_order",
           [](IR &ir, IR::NodeRef n,
              const std::vector<std::pair<IR::VarRef, std::pair<int, int>>>
                  &order) {
             std::vector<std::pair<IR::VarRef, IR::LoopSize>> wrapped_order;
             for (const auto &o : order) {
               wrapped_order.emplace_back(
                   o.first, IR::LoopSize{o.second.first, o.second.second});
             }
             ir.set_order(n, wrapped_order);
           })
      .def("disable_reuse", &IR::disable_reuse)
      .def("enable_reuse", &IR::enable_reuse)
      .def("dump", &IR::dump)
      .def("dump_var", [](IR &ir, IR::VarRef v) { return ir.var(v).name(); })
      .def_property_readonly("vars", &IR::vars)
      .def_property_readonly("nodes", &IR::nodes)
      .def_property_readonly(
          "order",
          [](IR &ir) {
            std::unordered_map<
                IR::NodeRef,
                std::vector<std::pair<IR::VarRef, std::pair<int, int>>>>
                order;
            for (const auto &n : ir.nodes()) {
              for (const auto &o : ir.order(n)) {
                order[n].emplace_back(std::make_pair(
                    o.first, std::make_pair(o.second.size, o.second.tail)));
              }
            }
            return order;
          })
      .def("pointwise_vars", &IR::pointwise_vars)
      .def("loop_vars", &IR::loop_vars)
      .def("output_vars",
           [](IR &ir, IR::NodeRef n) { return ir.node(n).vars(); });
  py::class_<Compiled>(m, "Compiled")
      .def(
          "__call__",
          [](const Compiled &cc, std::vector<std::shared_ptr<Tensor>> tensors,
             bool sync) {
            std::vector<void *> memory;
            for (auto &t : tensors) {
              ASSERT((t->data.compatible & cc.hardware_requirement) ==
                     cc.hardware_requirement)
                  << "Tensor on wrong hardware, perhaps use "
                     "lt.set_default_hardware(...)";
              memory.emplace_back(t->data.address);
            }
            cc.run(memory, sync);
          },
          py::arg("tensors"), py::arg("sync") = true)
      .def("__getattr__", [](const Compiled &cc, std::string name) {
        if (cc.int_properties.count(name)) {
          return py::cast(cc.int_properties.at(name));
        }
        if (cc.string_properties.count(name)) {
          return py::cast(cc.string_properties.at(name));
        }
        ASSERT(0) << "Couldn't find property " << name << " in " << cc.name
                  << " Compiled object";
        return py::cast(0);
      });
  for (auto &backend_pair : getBackends()) {
    auto &backend = backend_pair.second;
    if (backend->hardware_requirement() & getAvailableHardware()) {
      m.def(
          backend->name().c_str(),
          [=](const LoopTree &lt,
              const std::unordered_set<LoopTree::TreeRef> &parallel,
              LoopTree::TreeRef root) {
            return backend->compile(lt, parallel, root).release();
          },
          py::arg("loop_tree"),
          py::arg("parallel") = std::unordered_set<LoopTree::TreeRef>{},
          py::arg("root") = -1);
    }
  }
  py::class_<LoopTree>(m, "LoopTree")
      .def(py::init<const IR &>())
      .def("annotate", [](LoopTree &lt, LoopTree::TreeRef ref,
                          std::string annot) { lt.annotate(ref, annot); })
      .def_property_readonly("roots",
                             [](const LoopTree &lt) { return lt.roots; })
      .def("trivially_parallel",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             return trivially_parallel(lt, ref);
           })
      .def("children",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             return lt.tree_node(ref).children;
           })
      .def("leaves",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             std::vector<LoopTree::TreeRef> leaves;
             lt.walk(
                 [&](LoopTree::TreeRef r, int) {
                   if (lt.kind(r) == LoopTree::NODE) {
                     leaves.emplace_back(r);
                   }
                 },
                 ref);
             return leaves;
           })
      .def_property_readonly("loops",
                             [](const LoopTree &lt) {
                               std::vector<LoopTree::TreeRef> loops;
                               lt.walk([&](LoopTree::TreeRef r, int) {
                                 if (lt.kind(r) == LoopTree::LOOP) {
                                   loops.emplace_back(r);
                                 }
                               });
                               return loops;
                             })
      .def("ir_node",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             ASSERT(lt.kind(ref) == LoopTree::NODE);
             return lt.node(ref);
           })
      .def(
          "dump", &LoopTree::dump,
          py::arg("callback") = std::function<std::string(LoopTree::TreeRef)>{})
      .def(
          "__repr__", &LoopTree::dump,
          py::arg("callback") = std::function<std::string(LoopTree::TreeRef)>{})
      .def("__call__", [](const LoopTree &lt,
                          std::vector<std::shared_ptr<Tensor>> tensors) {
        std::vector<void *> memory;
        for (auto &t : tensors) {
          memory.emplace_back(t->data.address);
        }
        return exec(lt, memory);
      });

  py::class_<lazy::Tensor>(m, "Tensor").def(py::init([](size_t N) {
    return lazy::Tensor(N);
  }));

  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "RawTensor")
      .def(py::init([](size_t N, std::string hardware) {
             int hardware_id = -1;
             if (hardware == "default") {
               hardware_id = default_hardware_id;
             } else {
               for (auto &hw : getHardware()) {
                 if (hw->name() == hardware) {
                   hardware_id = hw->id();
                 }
               }
             }
             ASSERT(hardware_id >= 0)
                 << "Unregistered hardware name: " << hardware
                 << " (check available devices)";
             return std::make_shared<Tensor>(N, hardware_id);
           }),
           py::arg("size"), py::arg("hardware") = std::string("default"))
      .def("set",
           [](Tensor &t, float f) {
             auto data = (float *)t.data.address;
             for (auto i = 0; i < t.numel; ++i) {
               data[i] = f;
             }
#ifdef ENABLE_CUDA
             if (cuda_available) {
               CULIB(cuCtxSynchronize)();
             }
#endif
           })
      .def("set",
           [](Tensor &t, std::vector<float> fs) {
             auto data = (float *)t.data.address;
             ASSERT(fs.size() == t.numel);
             for (auto i = 0; i < t.numel; ++i) {
               data[i] = fs[i];
             }
#ifdef ENABLE_CUDA
             if (cuda_available) {
               CULIB(cuCtxSynchronize)();
             }
#endif
           })
      .def("set",
           [](Tensor &t,
              py::array_t<float, py::array::c_style | py::array::forcecast>
                  array) {
             py::buffer_info buf = array.request();
             ASSERT(buf.size == t.numel);
             float *data = static_cast<float *>(buf.ptr);
             float *tensor_data = (float *)t.data.address;
             for (auto i = 0; i < t.numel; ++i) {
               tensor_data[i] = data[i];
             }
#ifdef ENABLE_CUDA
             if (cuda_available) {
               CULIB(cuCtxSynchronize)();
             }
#endif
           })
      .def("to_numpy",
           [](Tensor &t) {
#ifdef ENABLE_CUDA
             if (cuda_available) {
               CULIB(cuCtxSynchronize)();
             }
#endif
             auto result = py::array_t<float>(t.numel);
             py::buffer_info buf = result.request();
             float *data = static_cast<float *>(buf.ptr);
             float *tensor_data = (float *)t.data.address;
             for (auto i = 0; i < t.numel; ++i) {
               data[i] = tensor_data[i];
             }
             return result;
           })
      .def("randn",
           [](Tensor &t, float mean, float stddev) {
             std::random_device rd{};
             std::mt19937 gen{rd()};
             std::normal_distribution<> d{mean, stddev};
             float *tensor_data = (float *)t.data.address;
             for (auto i = 0; i < t.numel; ++i) {
               tensor_data[i] = d(gen);
             }
           })
      .def("dump",
           [](const Tensor &t) {
             std::stringstream ss;
             float *tensor_data = (float *)t.data.address;
             ss << "numel: " << t.numel << ", [";
             if (t.numel < 9) {
               for (auto i = 0; i < t.numel; ++i) {
                 ss << tensor_data[i];
                 if (i + 1 != t.numel) {
                   ss << ", ";
                 }
               }
             } else {
               for (auto i = 0; i < 4; ++i) {
                 ss << tensor_data[i] << ", ";
               }
               ss << "..., ";
               for (auto i = t.numel - 4; i < t.numel; ++i) {
                 ss << tensor_data[i];
                 if (i + 1 != t.numel) {
                   ss << ", ";
                 }
               }
             }
             ss << "]";
             return ss.str();
           })
      .def("numel", [](Tensor &t) { return t.numel; });
}
