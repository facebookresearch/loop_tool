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
namespace py = pybind11;

#include "compile.h"
#include "error.h"
#include "ir.h"

#ifdef ENABLE_CUDA
#include "../backends/cuda/cuda_backend.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

struct Tensor {
  Tensor(int N) {
    void *ptr = nullptr;
    auto s = N * sizeof(float);
#ifdef ENABLE_CUDA
    auto err = cudaMallocManaged(&ptr, s); // N * sizeof(float));
    gpuErrchk(err);
#else
    ptr = malloc(s);
#endif
    data = (float *)ptr;
    numel = N;
  }
  ~Tensor() {
#ifndef ENABLE_CUDA
    free(data);
#endif
  }
  Tensor() = delete;
  float *data;
  size_t numel;
};

PYBIND11_MODULE(loop_tool_py, m) {
  py::class_<IR>(m, "IR")
      .def(py::init<>())
      .def("create_var", &IR::create_var)
      .def("create_node", &IR::create_node)
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
      .def("all_vars", &IR::all_vars)
      .def("output_vars",
           [](IR &ir, IR::NodeRef n) { return ir.node(n).vars(); });
#ifdef ENABLE_CUDA
  py::class_<CompiledCuda>(m, "CompiledCuda")
      .def(py::init<const LoopTree &,
                    const std::unordered_set<LoopTree::TreeRef> &>(),
           py::arg("loop_tree"),
           py::arg("threaded") = std::unordered_set<LoopTree::TreeRef>{-1})
      .def(
          "__call__",
          [](const CompiledCuda &cc, std::vector<Tensor> tensors, bool sync) {
            std::vector<void *> memory;
            for (auto &t : tensors) {
              memory.emplace_back(t.data);
            }
            cc(memory, sync);
          },
          py::arg("tensors"), py::arg("sync") = true)
      .def_property_readonly("code",
                             [](const CompiledCuda &cc) { return cc.code; })
      .def_property_readonly(
          "num_threads", [](const CompiledCuda &cc) { return cc.num_threads; })
      .def_property_readonly(
          "num_blocks", [](const CompiledCuda &cc) { return cc.num_blocks; })
      .def_property_readonly("bandwidth", [](const CompiledCuda &cc) {
        return cc.peak_bandwidth_gb;
      });
#endif
  py::class_<LoopTree>(m, "LoopTree")
      .def(py::init<const IR &>())
      .def_property_readonly("roots",
                             [](const LoopTree &lt) { return lt.roots; })
      .def("trivially_parallel",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             return trivially_parallel(lt, ref);
           })
      .def("children",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             return lt.node(ref).children;
           })
      .def("leaves",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             std::vector<LoopTree::TreeRef> leaves;
             lt.walk(
                 [&](LoopTree::TreeRef r, int) {
                   if (lt.node(r).kind == LoopTree::NODE) {
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
                                 if (lt.node(r).kind == LoopTree::LOOP) {
                                   loops.emplace_back(r);
                                 }
                               });
                               return loops;
                             })
      .def("ir_node",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             ASSERT(lt.node(ref).kind == LoopTree::NODE);
             return lt.node(ref).node;
           })
      .def("dump", &LoopTree::dump,
           py::arg("callback") =
               std::function<std::string(LoopTree::TreeRef)>{})
      .def("__repr__", &LoopTree::dump,
           py::arg("callback") =
               std::function<std::string(LoopTree::TreeRef)>{})
      .def("exec_cpu", [](const LoopTree &lt, std::vector<Tensor> tensors) {
        std::vector<void *> memory;
        for (auto &t : tensors) {
          memory.emplace_back(t.data);
        }
        return exec(lt, memory);
      });

  py::class_<Tensor>(m, "Tensor")
      .def(py::init<int>())
      .def("set",
           [](Tensor &t, float f) {
             for (auto i = 0; i < t.numel; ++i) {
               t.data[i] = f;
             }
#ifdef ENABLE_CUDA
             cuCtxSynchronize();
#endif
           })
      .def("set",
           [](Tensor &t, std::vector<float> fs) {
             ASSERT(fs.size() == t.numel);
             for (auto i = 0; i < t.numel; ++i) {
               t.data[i] = fs[i];
             }
#ifdef ENABLE_CUDA
             cuCtxSynchronize();
#endif
           })
      .def("set",
           [](Tensor &t,
              py::array_t<float, py::array::c_style | py::array::forcecast>
                  array) {
             py::buffer_info buf = array.request();
             ASSERT(buf.size == t.numel);
             float *data = static_cast<float *>(buf.ptr);
             for (auto i = 0; i < t.numel; ++i) {
               t.data[i] = data[i];
             }
#ifdef ENABLE_CUDA
             cuCtxSynchronize();
#endif
           })
      .def("to_numpy",
           [](Tensor &t) {
#ifdef ENABLE_CUDA
             cuCtxSynchronize();
#endif
             auto result = py::array_t<float>(t.numel);
             py::buffer_info buf = result.request();
             float *data = static_cast<float *>(buf.ptr);
             for (auto i = 0; i < t.numel; ++i) {
               data[i] = t.data[i];
             }
             return result;
           })
      .def("randn",
           [](Tensor &t, float mean, float stddev) {
             std::random_device rd{};
             std::mt19937 gen{rd()};
             std::normal_distribution<> d{mean, stddev};
             for (auto i = 0; i < t.numel; ++i) {
               t.data[i] = d(gen);
             }
           })
      .def("dump",
           [](const Tensor &t) {
             std::stringstream ss;
             ss << "numel: " << t.numel << ", [";
             if (t.numel < 9) {
               for (auto i = 0; i < t.numel; ++i) {
                 ss << t.data[i];
                 if (i + 1 != t.numel) {
                   ss << ", ";
                 }
               }
             } else {
               for (auto i = 0; i < 4; ++i) {
                 ss << t.data[i] << ", ";
               }
               ss << "..., ";
               for (auto i = t.numel - 4; i < t.numel; ++i) {
                 ss << t.data[i];
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
