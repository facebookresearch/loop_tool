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
#define LOCATION __FILE__ " : " S2(__LINE__)

class NullStream : public std::ostream {
public:
  NullStream() : std::ostream(nullptr) {}
  NullStream(const NullStream &) : std::ostream(nullptr) {}
};

template <class T>
const NullStream &operator<<(NullStream &&os, const T &value) {
  return os;
}

struct StreamOut {
  std::stringstream ss;
  bool failure = false;

  StreamOut(bool pass, std::string location, std::string cond = "")
      : failure(!pass) {
    if (failure && cond.size()) {
      ss << "assertion: " << cond << " ";
    }
    ss << "failed @ " << location << " ";
  }

  template <typename T> StreamOut &operator<<(const T &d) {
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

#ifdef NOEXCEPTIONS
#define ASSERT(x) assert(x)
#else
#define ASSERT(x) StreamOut(x, LOCATION, #x)
#endif
