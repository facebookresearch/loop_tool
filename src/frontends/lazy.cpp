#include "loop_tool/lazy.h"
namespace loop_tool {
namespace lazy {

const int getNewSymbolId() {
  static int symbol_count_ = 0;
  return symbol_count_++;
}

}  // namespace lazy
}  // namespace loop_tool
