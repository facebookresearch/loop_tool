#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>

#include "test_utils.h"

namespace loop_tool {
namespace testing {

AddTest::AddTest(std::string file, std::string name, std::function<void()> fn) {
  getTestRegistry().emplace_back(file, name, fn);
}

std::vector<Test>& getTestRegistry() {
  static std::vector<Test> tests_;
  return tests_;
}

#define ANSI_GREEN "\033[32m"
#define ANSI_RED "\033[31m"
#define ANSI_RESET "\033[39m"

void runner(int argc, char* argv[]) {
  // unsigned int microseconds = 1000;
  // usleep(microseconds);
  bool verbose = false;
  bool strict = false;
  std::regex filter(".*");
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
    if (arg == "--filter") {
      if (argc <= i + 1) {
        std::cerr << "no argument found for --filter\n";
        return;
      }
      arg = argv[++i];
      filter = std::regex(arg);
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
    return a.file.compare(b.file) < 0;
  });
  tests.erase(std::remove_if(tests.begin(), tests.end(),
                             [&](const Test& test) {
                               std::string q = test.file + " " + test.name;
                               return !std::regex_search(q, filter);
                             }),
              tests.end());
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
      C[m * N + n] = (alpha ? (alpha * C[m * N + n]) : 0) + tmp;
    }
  }
}

void ref_conv(const float* X, const float* W, int N, int M, int C, int HW,
              int K, float* Y) {
  const auto HWO = HW - K + 1;
  for (auto i = 0; i < N * M * HWO * HWO; ++i) {
    Y[i] = 0;
  }
  for (auto n = 0; n < N; ++n) {
    for (auto m = 0; m < M; ++m) {
      for (auto c = 0; c < C; ++c) {
        for (auto h = 0; h < HWO; ++h) {
          for (auto w = 0; w < HWO; ++w) {
            for (auto kh = 0; kh < K; ++kh) {
              for (auto kw = 0; kw < K; ++kw) {
                Y[n * M * HWO * HWO + m * HWO * HWO + h * HWO + w] +=
                    X[(n)*HW * HW * C + (c)*HW * HW + (h + kh) * HW +
                      (w + kw)] *
                    W[m * C * K * K + c * K * K + kh * K + kw];
              }
            }
          }
        }
      }
    }
  }
}

bool all_close(const float* A, const float* B, size_t N, float eps) {
  float max_diff = 0;
  float min_val = std::numeric_limits<float>::max();
  for (size_t i = 0; i < N; ++i) {
    max_diff = std::max(std::abs(A[i] - B[i]), max_diff);
    min_val = std::min(std::abs(A[i]), min_val);
    min_val = std::min(std::abs(B[i]), min_val);
  }
  std::cerr << "max diff " << max_diff << " vs min val " << min_val
            << " (eps: " << eps << ")\n";
  return max_diff < std::max(eps * min_val, eps);
}

}  // namespace testing
}  // namespace loop_tool
