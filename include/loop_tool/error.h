/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define S1(x) #x
#define S2(x) S1(x)
#define LOCATION __FILE__ ":" S2(__LINE__)

namespace loop_tool {

struct NullStream {
  template <typename T>
  NullStream &operator<<(T const &) {
    return *this;
  }
};

struct StreamOut : public NullStream {
  std::stringstream ss;
  bool failure = false;

  StreamOut(bool pass, std::string location, std::string cond = "")
      : failure(!pass) {
    if (failure && cond.size()) {
      ss << "assertion: " << cond << " ";
    }
    ss << "failed @ " << location << " ";
  }

  template <typename T>
  StreamOut &operator<<(const T &d) {
    if (failure) {
      ss << d;
    }
    return *this;
  }

  ~StreamOut() noexcept(false) {
    if (failure) {
      throw std::runtime_error(ss.str());
    }
  }
};

}  // namespace loop_tool

#ifdef NOEXCEPTIONS
#define ASSERT(x) loop_tool::NullStream()
#else
#define ASSERT(x) \
  if (!(x)) loop_tool::StreamOut(x, LOCATION, #x)
#endif
