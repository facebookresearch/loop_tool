#pragma once

#include <functional>
#include <vector>

namespace loop_tool {
namespace testing {

struct Test {
  std::string file;
  std::string name;
  std::function<void()> fn;
  Test(std::string file_, std::string name_, std::function<void()> fn_)
      : file(file_), name(name_), fn(fn_) {}
  void operator()() const { fn(); }
};

std::vector<Test> &getTestRegistry();

struct AddTest {
  AddTest(std::string file, std::string name, std::function<void()> fn) {
    getTestRegistry().emplace_back(file, name, fn);
  }
};

void runner(int argc, char *argv[]);
void rand(float *data, int N);
void ref_mm(const float *A, const float *B, int M, int N, int K, float *C,
            float alpha = 0);

}  // namespace testing
}  // namespace loop_tool

#define TEST(name)                                               \
  void _loop_tool_test_##name();                                 \
  static loop_tool::testing::AddTest _loop_tool_test_add_##name( \
      __FILE__, #name, _loop_tool_test_##name);                  \
  void _loop_tool_test_##name()

#define RUN_TESTS(argc, argv) loop_tool::testing::runner(argc, argv);