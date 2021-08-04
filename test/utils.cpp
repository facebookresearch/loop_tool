#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>

#include "test_utils.h"

namespace loop_tool {
namespace testing {

std::vector<Test>& getTestRegistry() {
  static std::vector<Test> tests_;
  return tests_;
}

#define ANSI_GREEN "\033[32m"
#define ANSI_RED "\033[31m"
#define ANSI_RESET "\033[39m"

void runner(int argc, char* argv[]) {
  bool verbose = false;
  bool strict = false;
  for (auto i = 0; i < argc; ++i) {
    auto arg = std::string(argv[i]);
    if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    }
    if (arg == "--strict" || arg == "-f") {
      strict = true;
    }
    if (arg == "-fv" || arg == "-vf") {
      strict = true;
      verbose = true;
    }
  }
  std::stringstream stdout_buffer;
  std::stringstream stderr_buffer;
  std::streambuf* old_stdout;
  std::streambuf* old_stderr;
  auto hide_output = [&]() {
    stdout_buffer.str("");
    stderr_buffer.str("");
    if (!verbose) {
      old_stdout = std::cout.rdbuf(stdout_buffer.rdbuf());
      old_stderr = std::cerr.rdbuf(stderr_buffer.rdbuf());
    }
  };
  auto restore_output = [&]() {
    if (!verbose) {
      std::cout.rdbuf(old_stdout);
      std::cerr.rdbuf(old_stderr);
    }
  };

  auto tests = getTestRegistry();
  std::sort(tests.begin(), tests.end(), [](const Test& a, const Test& b) {
    return a.file.compare(b.file);
  });
  std::string curr_file = "";
  size_t passed = 0;

  for (const auto& test : tests) {
    if (test.file != curr_file) {
      std::cout << "running tests in " << test.file << "\n";
      ;
      curr_file = test.file;
    }

    std::cout << " - " << test.name << " ... ";

    try {
      hide_output();
      test();
      restore_output();
      std::cout << ANSI_GREEN << "passed" << ANSI_RESET << ".\n";
      passed++;
    } catch (const std::exception& e) {
      restore_output();
      std::cout << ANSI_RED << "failed" << ANSI_RESET << ".\n";
      if (strict) {
        throw;
      } else {
        if (stdout_buffer.str().size()) {
          std::cout << "==== stdout for failed test \"" << test.name
                    << "\" ====\n";
          std::cout << stdout_buffer.str();
          std::cout << "\n[ run tests with -f flag to throw ]\n";
        }
        if (stderr_buffer.str().size()) {
          std::cerr << "==== stderr for failed test \"" << test.name
                    << "\" ====\n";
          std::cerr << stderr_buffer.str();
          std::cerr << "\n[ run tests with -f flag to throw ]\n";
        }
      }
    }
  }
  std::cout << "[" << passed << "/" << tests.size() << " tests passed]\n";
}

void rand(float* data, int N) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::normal_distribution<> dist(2, 2);
  for (auto i = 0; i < N; ++i) {
    data[i] = dist(e2);
  }
}

// assumes LCA=K, LCB=N
void ref_mm(const float* A, const float* B, int M, int N, int K, float* C,
            float alpha) {
  for (auto n = 0; n < N; ++n) {
    for (auto m = 0; m < M; ++m) {
      float tmp = 0;
      for (auto k = 0; k < K; ++k) {
        tmp += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = alpha * C[m * N + n] + tmp;
    }
  }
}

}  // namespace testing
}  // namespace loop_tool
