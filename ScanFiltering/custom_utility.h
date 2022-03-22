#pragma once

#include <string>
#include <type_traits>

namespace cu {
template <class T>
constexpr std::underlying_type<T>::type to_underlying_type(T v) {
  return static_cast<std::underlying_type_t<T>>(v);
}
constexpr auto tot(auto v) { return to_underlying_type(v); }
void remove_ru_separator(std::string &str);
} // namespace cu
