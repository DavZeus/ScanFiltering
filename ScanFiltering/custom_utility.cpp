#include "custom_utility.h"

namespace cu {
void remove_wrong_ru_separator(std::string &str) {
  size_t pos = 0;
  char wrong_separator[] = "Â";
  size_t found_pos{};
  while (found_pos = str.find(wrong_separator, pos) != std::string::npos) {
    str.erase(found_pos, sizeof wrong_separator);
    pos = found_pos;
  }
}
} // namespace cu
