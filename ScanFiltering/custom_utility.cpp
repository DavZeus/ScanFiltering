#include "custom_utility.h"

namespace cu {
void remove_wrong_ru_separator(std::string &str) {
  size_t pos = 0;
  char wrong_separator[] = "Â";
  while (auto found_pos = str.find(wrong_separator, pos)) {
    str.erase(found_pos, sizeof wrong_separator);
    pos = found_pos;
  }
}
} // namespace cu
