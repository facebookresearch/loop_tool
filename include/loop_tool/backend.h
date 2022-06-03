/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "ir.h"
#include "tensor.h"

namespace loop_tool {

struct Compiled {
  virtual ~Compiled() {}
  virtual void run(const std::vector<void *> &memory,
                   bool sync = true) const = 0;
  virtual std::string dump() const {
    return "[not implemented, override `std::string Compiled::dump() const`]";
  }

  void operator()(const std::vector<Tensor *> &tensors, bool sync = true) const;

  std::vector<void *> allocate(std::vector<int64_t> &sizes) const;

  template <bool sync, typename... Args>
  void run(Args const &...tensors) const {
    std::vector<void *> memory = {tensors.data.address...};
    run(memory, sync);
  }

  template <typename... Args>
  void operator()(Args const &...tensors) const {
    run<true, Args...>(std::forward<Args const &>(tensors)...);
  }

  template <typename... Args>
  void async(Args const &...tensors) const {
    run<false, Args...>(std::forward<Args const &>(tensors)...);
  }

  std::unordered_map<std::string, int> int_properties;
  std::unordered_map<std::string, std::string> string_properties;
  int hardware_requirement = -1;
  std::string name;
};

struct Backend {
  std::string name_;
  Backend(std::string name) : name_(name) {}
  virtual ~Backend(){};

  const std::string &name() const { return name_; }

  virtual std::unique_ptr<Compiled> compile_impl(const LoopTree &lt) const = 0;
  virtual int hardware_requirement() const = 0;

  std::unique_ptr<Compiled> compile(const LoopTree &lt) const {
    auto compiled = compile_impl(lt);
    compiled->hardware_requirement = hardware_requirement();
    compiled->name = name();
    return compiled;
  }
};

const std::unordered_map<std::string, std::shared_ptr<Backend>> &getBackends();
void registerBackend(std::shared_ptr<Backend> backend);

std::shared_ptr<Backend> &getDefaultBackend();
void setDefaultBackend(std::string backend);

struct ScopedBackend {
  std::string old_backend_name;
  ScopedBackend(std::string backend_name) {
    const auto &old_backend = getDefaultBackend();
    old_backend_name = old_backend->name();
    setDefaultBackend(backend_name);
  }
  ~ScopedBackend() { setDefaultBackend(old_backend_name); }
};

struct RegisterBackend {
  RegisterBackend(std::shared_ptr<Backend> backend) {
    registerBackend(backend);
  }
};

void loadLibrary(std::string lib_name);

#ifndef DABUN_ISA

#if defined(__AVX512F__)
#define DABUN_ISA avx512
#elif defined(__aarch64__)
#define DABUN_ISA aarch64
#else  // default to avx2
// #elif defined(__AVX2__)
#define DABUN_ISA avx2
// #error "ISA not supported"
#endif

#endif

struct avx2 {};
struct avx512 {};
struct avx2_plus {};
struct aarch64 {};

template <class>
struct isa_traits;

template <>
struct isa_traits<avx2> {
  static constexpr int total_vector_registers = 16;
  static constexpr int vector_register_mask = 1;
  static constexpr int vector_size = 8;
};

template <>
struct isa_traits<avx512> {
  static constexpr int total_vector_registers = 32;
  static constexpr int vector_register_mask = 0;
  static constexpr int vector_size = 16;
};

template <>
struct isa_traits<avx2_plus> {
  static constexpr int total_vector_registers = 16;
  static constexpr int vector_register_mask = 0;
  static constexpr int vector_size = 8;
};

template <>
struct isa_traits<aarch64> {
  static constexpr int total_vector_registers = 32;
  static constexpr int vector_register_mask = 0;
  static constexpr int vector_size = 4;
  static constexpr int fp16_vector_size = 2;
};

}  // namespace loop_tool
