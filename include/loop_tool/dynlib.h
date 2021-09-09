/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once
#include <dlfcn.h>

namespace loop_tool {

struct DynamicLibrary {
  DynamicLibrary(const char* name) : name_(name) {
    lib_ = dlopen(name, RTLD_LOCAL | RTLD_NOW);
    ASSERT(lib_) << "Couldn't load library " << name_;
  }

  static bool exists(const char* name) {
    return !!dlopen(name, RTLD_LOCAL | RTLD_NOW);
  }

  inline void* sym(const char* sym_name) const {
    ASSERT(lib_) << "Library " << name_ << " not loaded for symbol "
                 << sym_name;
    auto* symbol = dlsym(lib_, sym_name);
    ASSERT(symbol) << "Couldn't find " << sym_name << " in " << name_;
    return symbol;
  }

  ~DynamicLibrary() { dlclose(lib_); }

 private:
  void* lib_ = nullptr;
  std::string name_;
};

#define DYNLIB(lib, name) reinterpret_cast<decltype(&name)>(lib->sym(#name))

}  // namespace loop_tool
