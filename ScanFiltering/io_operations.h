#pragma once

#include <filesystem>
#include <string>

namespace sf {
namespace io {
std::string generate_time_string();
std::filesystem::path make_save_folder(std::string name = "");
} // namespace io
} // namespace sf
