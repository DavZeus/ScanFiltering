#include "io_operations.h"

#include <chrono>
#include <filesystem>
#include <sstream>

namespace sf {
namespace io {
std::string generate_time_string() {
  std::stringstream time_parse;
  tm tm;
  auto time =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  localtime_s(&tm, &time);
  time_parse << std::put_time(&tm, "%FT%H-%M-%S") << std::flush;
  return time_parse.str();
}
std::filesystem::path make_save_folder(std::string name) {
  std::filesystem::path path{name};
  if (path.empty()) {
    path = generate_time_string();
  } else {
    path += "-" + generate_time_string();
  }
  std::filesystem::create_directory(path);
  return path;
}
} // namespace io
} // namespace sf
