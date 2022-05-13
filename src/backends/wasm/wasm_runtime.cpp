/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/wasm.h"
#include "wasm_c_api.h"

using namespace loop_tool;

struct WebAssemblyCompiled : public Compiled {
  std::vector<uint8_t> emitted_wasm;
  wasm_engine_t* engine;
  wasm_store_t* store;
  wasm_instance_t* instance;
  wasm_extern_vec_t exports;
  wasm_memory_t* wasm_memory;
  wasm_func_t* fn;

  std::vector<int64_t> memory_size_map;
  std::vector<int64_t> memory_input_offset_map;
  std::vector<int64_t> memory_output_offset_map;

  WebAssemblyCompiled(const LoopTree& lt) {
    WebAssemblyCompiler wc(lt);
    emitted_wasm = wc.emit();
    engine = wasm_engine_new();
    store = wasm_store_new(engine);
    wasm_byte_vec_t binary;
    wasm_byte_vec_new_uninitialized(&binary, emitted_wasm.size());
    memcpy(binary.data, emitted_wasm.data(), emitted_wasm.size());
    wasm_module_t* m = wasm_module_new(store, &binary);
    ASSERT(m) << "Couldn't compile WebAssembly module";
    wasm_byte_vec_delete(&binary);

    wasm_extern_vec_t imports = WASM_EMPTY_VEC;
    instance =
        wasm_instance_new_with_args(store, m, &imports, NULL, KILOBYTE(32), 0);
    ASSERT(instance) << "Couldn't instantiate WebAssembly module";
    wasm_instance_exports(instance, &exports);
    wasm_memory = wasm_extern_as_memory(exports.data[0]);
    fn = wasm_extern_as_func(exports.data[1]);
    wasm_module_delete(m);

    // include input/output sizes
    const auto& inputs = lt.ir.inputs();
    const auto& outputs = lt.ir.outputs();
    auto all_mem_sizes = wc.memory_sizes(true);
    int64_t offset = 0;
    memory_input_offset_map.resize(inputs.size() + outputs.size());
    memory_output_offset_map.resize(inputs.size() + outputs.size());
    for (auto i = 0; i < inputs.size() + outputs.size(); ++i) {
      int64_t size = all_mem_sizes.at(i) * 4;
      memory_size_map.emplace_back(size);
      if (i < inputs.size()) {
        memory_input_offset_map[i] = offset;
        memory_output_offset_map[i] = -1;
      } else {
        memory_input_offset_map[i] = -1;
        memory_output_offset_map[i] = offset;
      }
      offset += size;
    }
  }

  void run(const std::vector<void*>& user_memory, bool sync) const override {
    wasm_val_vec_t args = WASM_EMPTY_VEC;
    wasm_val_vec_t results = WASM_EMPTY_VEC;
    // copy user memory to webassembly and back
    char* data = wasm_memory_data(wasm_memory);
    for (auto i = 0; i < user_memory.size(); ++i) {
      int64_t offset = memory_input_offset_map[i];
      if (offset == -1) {
        continue;
      }
      void* ptr = &data[offset];
      memcpy(ptr, user_memory[i], memory_size_map[i]);
    }
    wasm_func_call(fn, &args, &results);
    for (auto i = 0; i < user_memory.size(); ++i) {
      int64_t offset = memory_output_offset_map[i];
      if (offset == -1) {
        continue;
      }
      void* ptr = &data[offset];
      memcpy(user_memory[i], ptr, memory_size_map[i]);
    }
  }
  std::string dump() const override { return ""; }

  ~WebAssemblyCompiled() {
    wasm_extern_vec_delete(&exports);
    wasm_instance_delete(instance);
    wasm_store_delete(store);
    wasm_engine_delete(engine);
  }
};

struct WebAssemblyBackend : public Backend {
  WebAssemblyBackend() : Backend("wasm") {}
  ~WebAssemblyBackend() {}
  WebAssemblyBackend(std::string name) : Backend(name) {}

  std::unique_ptr<Compiled> compile_impl(const LoopTree& lt) const override {
    return std::make_unique<WebAssemblyCompiled>(lt);
  }
  int hardware_requirement() const override { return 1 << 0; }
};

static RegisterBackend wasm_backend_reg_(
    std::make_shared<WebAssemblyBackend>());
