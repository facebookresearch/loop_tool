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
#include "loop_tool/mutate.h"
#include "loop_tool/tensor.h"

#ifdef ENABLE_CUDA
#include <cuda.h>

#include "cuda_backend.h"
#endif

using namespace loop_tool;
namespace py = pybind11;

static int default_hardware_id = 0;  // CPU
static bool cuda_available = false;

class SymbolGenerator {
 public:
  SymbolGenerator &enter() { return *this; }
  void exit(const py::object &type, const py::object &value,
            const py::object &traceback) {}
  symbolic::Symbol getSymbol(std::string s) {
    if (!symbols_.count(s)) {
      symbols_.emplace(s, symbolic::Symbol(s));
    }
    return symbols_.at(s);
  }

 private:
  std::unordered_map<std::string, symbolic::Symbol> symbols_;
};

PYBIND11_MODULE(loop_tool_py, m) {
  m.def("load_lib", [](std::string lib_name) { loadLibrary(lib_name); });
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
    for (const auto &hw : getHardware()) {
      if (hw->name() == hardware) {
        default_hardware_id = hw->id();
        setDefaultHardwareId(hw->id());
        set = true;
      }
    }
    ASSERT(set) << "cannot find hardware: " << hardware;
  });
  m.def("set_default_backend",
        [](std::string backend) { setDefaultBackend(backend); });
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
           py::arg("constraints") = std::vector<symbolic::Constraint>{},
           py::arg("sym_var_map") = std::unordered_map<int, IR::VarRef>{})
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
      .def("__repr__", &dot)
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
  py::class_<Compiled, std::shared_ptr<Compiled>>(m, "Compiled")
      .def(
          "__call__",
          [](Compiled *cc, std::vector<std::shared_ptr<Tensor>> tensors,
             bool sync) {
            std::vector<void *> memory;
            for (const auto &t : tensors) {
              ASSERT((t->data.compatible & cc->hardware_requirement) ==
                     cc->hardware_requirement)
                  << "Tensor on wrong hardware, perhaps use "
                     "lt.set_default_hardware(...)";
              memory.emplace_back(t->data.address);
            }
            cc->run(memory, sync);
          },
          py::arg("tensors"), py::arg("sync") = true)
      .def("__getattr__", [](Compiled *cc, std::string name) {
        if (cc->int_properties.count(name)) {
          return py::cast(cc->int_properties.at(name));
        }
        if (cc->string_properties.count(name)) {
          return py::cast(cc->string_properties.at(name));
        }
        ASSERT(0) << "Couldn't find property " << name << " in " << cc->name
                  << " Compiled object";
        return py::cast(0);
      });
  for (const auto &backend_pair : getBackends()) {
    auto &backend = backend_pair.second;
    if (backend->hardware_requirement() & getAvailableHardware()) {
      m.def(
          backend->name().c_str(),
          [=](const LoopTree &lt,
              const std::unordered_set<LoopTree::TreeRef> &parallel,
              LoopTree::TreeRef root) -> std::shared_ptr<Compiled> {
            return backend->compile(lt, parallel, root);
          },
          py::arg("loop_tree"),
          py::arg("parallel") = std::unordered_set<LoopTree::TreeRef>{},
          py::arg("root") = -1);
    }
  }
  py::class_<Compiler::Allocation>(m, "CompilerAllocation")
      .def_property_readonly("size", &Compiler::Allocation::size);
  py::class_<Compiler>(m, "Compiler")
      .def(py::init<const LoopTree &>())
      .def_property_readonly("allocations",
                             [](const Compiler &c) { return c.allocations; });

  py::class_<LoopTree::Loop>(m, "Loop")
      .def_property_readonly("var",
                             [](LoopTree::Loop &loop) { return loop.var; })
      .def_property_readonly("size",
                             [](LoopTree::Loop &loop) { return loop.size; })
      .def_property_readonly("tail",
                             [](LoopTree::Loop &loop) { return loop.tail; })
      .def("__eq__", &LoopTree::Loop::operator==);

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
      .def("children", &LoopTree::children)
      .def("parent", &LoopTree::parent)
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
      .def_property_readonly("ir", [](const LoopTree &lt) { return lt.ir; })
      .def("ir_node",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             ASSERT(lt.kind(ref) == LoopTree::NODE);
             return lt.node(ref);
           })
      .def("loop",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             ASSERT(lt.kind(ref) == LoopTree::LOOP);
             return lt.loop(ref);
           })
      .def("is_loop",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             return lt.kind(ref) == LoopTree::LOOP;
           })
      .def("dump",
           [](const LoopTree &lt, LoopTree::TreeRef ref) {
             if (lt.kind(ref) == LoopTree::LOOP) {
               auto loop = lt.loop(ref);
               std::stringstream ss;
               ss << "L{" << lt.ir.var(loop.var).name() << ":" << loop.size;
               if (loop.tail) {
                 ss << "r" << loop.tail;
               }
               ss << "}";
               return ss.str();
             }
             return lt.ir.dump(lt.node(ref));
           })
      .def(
          "dump", &LoopTree::dump,
          py::arg("callback") = std::function<std::string(LoopTree::TreeRef)>{})
      .def("walk", &LoopTree::walk, py::arg("callback"), py::arg("root") = -1)
      .def(
          "__repr__", &LoopTree::dump,
          py::arg("callback") = std::function<std::string(LoopTree::TreeRef)>{})
      .def("__call__", [](const LoopTree &lt,
                          std::vector<std::shared_ptr<Tensor>> tensors) {
        std::vector<void *> memory;
        for (const auto &t : tensors) {
          memory.emplace_back(t->data.address);
        }
        return exec(lt, memory);
      });
  m.def("swap", [](LoopTree &lt, LoopTree::TreeRef a, LoopTree::TreeRef b) {
    return loop_tool::swap(lt, a, b);
  });
  m.def("split", [](LoopTree &lt, LoopTree::TreeRef ref, int64_t size) {
    return loop_tool::split(lt, ref, size);
  });

  py::class_<lazy::Symbol>(m, "Symbol")
      .def(py::init<std::string>())
      .def("__mul__",
           [](lazy::Symbol &s, lazy::Symbol &other) { return s * other; })
      .def("__add__",
           [](lazy::Symbol &s, lazy::Symbol &other) { return s + other; })
      .def("__mul__",
           [](lazy::Symbol &s, lazy::Expr &other) { return s * other; })
      .def("__add__",
           [](lazy::Symbol &s, lazy::Expr &other) { return s + other; })
      .def("__repr__", [](lazy::Symbol &s) { return lazy::Expr(s).dump(); });
  py::class_<lazy::Expr>(m, "Expr")
      .def(py::init<size_t>())
      .def("__mul__",
           [](lazy::Expr &s, lazy::Expr &other) { return s * other; })
      .def("__add__",
           [](lazy::Expr &s, lazy::Expr &other) { return s + other; })
      .def("__mul__",
           [](lazy::Expr &s, lazy::Symbol &other) { return s * other; })
      .def("__add__",
           [](lazy::Expr &s, lazy::Symbol &other) { return s + other; })
      .def("__repr__", [](lazy::Expr &e) { return e.dump(); });

  m.def("Size", [](lazy::Symbol &s) { return lazy::Expr::size(s); });

  py::class_<SymbolGenerator>(m, "SymbolGenerator")
      .def(py::init<>())
      .def("__enter__", &SymbolGenerator::enter)
      .def("__exit__", &SymbolGenerator::exit)
      .def("__getattr__", [](SymbolGenerator &s, std::string name) {
        return s.getSymbol(name);
      });

  py::class_<lazy::Tensor>(m, "Tensor")
      .def(py::init([](py::args args) {
        std::vector<size_t> sizes;
        std::vector<lazy::Symbol> shape;
        for (const auto &arg : args) {
          if ((std::string)py::str(arg.get_type()) ==
              "<class 'loop_tool_py.Symbol'>") {
            shape.emplace_back(py::cast<lazy::Symbol>(arg));
          } else {
            sizes.emplace_back(py::cast<size_t>(arg));
          }
        }
        ASSERT(!!sizes.size() ^ !!shape.size())
            << "cannot mix numeric and symbolic instantiation yet";
        if (sizes.size()) {
          return lazy::Tensor(sizes);
        }
        return lazy::Tensor(shape);
      }))
      .def("__mul__",
           [](lazy::Tensor &t, lazy::Tensor &other) { return t * other; })
      .def("__div__",
           [](lazy::Tensor &t, lazy::Tensor &other) { return t / other; })
      .def("__add__",
           [](lazy::Tensor &t, lazy::Tensor &other) { return t + other; })
      .def("__sub__",
           [](lazy::Tensor &t, lazy::Tensor &other) { return t - other; })
      .def("__or__",
           [](lazy::Tensor &t, lazy::Tensor &other) { return t | other; })
      .def("max",
           [](lazy::Tensor &t, lazy::Tensor &other) { return t.max(other); })
      .def("__neg__", [](lazy::Tensor &t) { return -t; })
      .def("reciprocal", [](lazy::Tensor &t) { return t.reciprocal(); })
      .def("exp", [](lazy::Tensor &t) { return t.exp(); })
      .def("pad", [](lazy::Tensor &t, lazy::Symbol s,
                     int64_t amt) { return t.pad(s, amt); })
      .def("transpose",
           [](lazy::Tensor &t, py::args args) {
             std::vector<lazy::Symbol> vars;
             for (const auto &arg : args) {
               vars.emplace_back(py::cast<lazy::Symbol>(arg));
             }
             return t.transpose(vars);
           })
      .def("sum",
           [](lazy::Tensor &t, py::args args) {
             std::vector<lazy::Symbol> vars;
             for (const auto &arg : args) {
               vars.emplace_back(py::cast<lazy::Symbol>(arg));
             }
             return t.sum(vars);
           })
      .def("to",
           [](lazy::Tensor &t, py::args args, const py::kwargs &kwargs) {
             std::vector<lazy::Symbol> output_shape;
             for (const auto &arg : args) {
               output_shape.emplace_back(py::cast<lazy::Symbol>(arg));
             }
             if (kwargs.size()) {
               ASSERT(kwargs.size() == 1);
               auto kw = *kwargs.begin();
               ASSERT(std::string(py::str(kw.first)) == "constraints");
               std::vector<lazy::Constraint> constraints;
               for (auto t : py::cast<py::list>(kw.second)) {
                 auto rhs = py::cast<lazy::Expr>(py::cast<py::tuple>(t)[1]);
                 auto constraint = [&]() -> lazy::Constraint {
                   auto arg = py::cast<py::tuple>(t)[0];
                   if ((std::string)py::str(arg.get_type()) ==
                       "<class 'loop_tool_py.Symbol'>") {
                     return std::make_pair(py::cast<lazy::Symbol>(arg), rhs);
                   }
                   return std::make_pair(py::cast<lazy::Expr>(arg), rhs);
                 }();
                 // auto sym =
                 // py::cast<lazy::Symbol>(py::cast<py::tuple>(t)[0]);
                 constraints.emplace_back(
                     constraint);  // std::make_pair(sym, expr));
               }
               return t.to(output_shape, constraints);
             }
             return t.as(output_shape);
           })
      .def("__getitem__",
           [](lazy::Tensor &t, py::args args_) {
             auto args = py::cast<py::tuple>(args_[0]);
             if (args.size() != t.shape().size()) {
               throw py::index_error("Expected shape of size " +
                                     std::to_string(t.shape().size()) +
                                     " got " + std::to_string(args.size()));
             }
             std::vector<lazy::Symbol> output_shape;
             std::unordered_set<lazy::Symbol, symbolic::Hash<lazy::Symbol>>
                 output_syms;
             std::vector<lazy::Constraint> constraints;
             auto i = 0;
             for (const auto &a : args) {
               auto expr = [&]() {
                 if ((std::string)py::str(a.get_type()) ==
                     "<class 'loop_tool_py.Symbol'>") {
                   return lazy::Expr(py::cast<lazy::Symbol>(a));
                 }
                 return py::cast<lazy::Expr>(a);
               }();
               expr.walk([&](const lazy::Expr &e) {
                 if (e.type() == lazy::Expr::Type::symbol) {
                   auto sym = e.symbol();
                   ;
                   if (output_syms.count(sym) == 0) {
                     output_shape.emplace_back(sym);
                     output_syms.insert(sym);
                   }
                 }
                 return e;
               });
               constraints.emplace_back(t.shape().at(i), expr);
               i++;
             }
             return t.to(output_shape, constraints);
           })
      .def("set",
           [](lazy::Tensor &t,
              py::array_t<float, py::array::c_style | py::array::forcecast>
                  array) -> lazy::Tensor & {
             py::buffer_info buf = array.request();
             ASSERT(!t.has_deps()) << "cannot set data to a computed tensor";
             size_t numel = t.numel();
             ASSERT(buf.size == numel);
             float *data = static_cast<float *>(buf.ptr);
             float *tensor_data = t.data<float>();
             for (auto i = 0; i < numel; ++i) {
               tensor_data[i] = data[i];
             }
             return t;
           })
      .def("set_size",
           [](lazy::Tensor &t, py::args args) {
             std::vector<size_t> sizes;
             for (const auto &arg : args) {
               sizes.emplace_back(py::cast<size_t>(arg));
             }
             t.bind(nullptr, sizes);
             return t;
           })
      .def("set",
           [](lazy::Tensor &t, const IR ir) -> lazy::Tensor & {
             t.set(ir);
             return t;
           })
      .def("set",
           [](lazy::Tensor &t, const LoopTree loop_tree) -> lazy::Tensor & {
             t.set(loop_tree);
             return t;
           })
      .def("numpy",
           [](lazy::Tensor &t) {
             size_t numel = t.numel();
             auto result = py::array_t<float>(t.sizes());
             py::buffer_info buf = result.request();
             float *data = static_cast<float *>(buf.ptr);
#ifdef ENABLE_CUDA
             if (cuda_available) {
               CULIB(cuCtxSynchronize)();
             }
#endif
             float *tensor_data = t.data<float>();
             for (auto i = 0; i < numel; ++i) {
               data[i] = tensor_data[i];
             }
             return result;
           })
      .def("unify", [](lazy::Tensor &t) { t.unify(); })
      .def("compile", [](lazy::Tensor &t) { t.compile(); })
      .def("resolve", [](lazy::Tensor &t) { (void)t.data<float>(); })
      .def_property_readonly("compiled",
                             [](lazy::Tensor &t) { return t.compiled(); })
      .def_property_readonly("ir", [](lazy::Tensor &t) { return t.ir(); })
      .def_property_readonly("loop_tree",
                             [](lazy::Tensor &t) { return t.loop_tree(); })
      .def_property_readonly("code", [](lazy::Tensor &t) { return t.code(); })
      .def_property_readonly("symbolic_shape",
                             [](lazy::Tensor &t) { return t.shape(); })
      .def_property_readonly("shape", [](lazy::Tensor &t) {
        std::vector<size_t> sizes;
        for (auto i = 0; i < t.shape().size(); ++i) {
          sizes.emplace_back(t.size(i));
        }
        return sizes;
      });

  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "RawTensor")
      .def(py::init([](size_t N, std::string hardware) {
             int hardware_id = -1;
             if (hardware == "default") {
               hardware_id = default_hardware_id;
             } else {
               for (const auto &hw : getHardware()) {
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
