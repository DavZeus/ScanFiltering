#include "image_processor.h"

#include <fmt/format.h>

namespace sf {
Mat image_processor::detect_edges(Mat img) {
  Mat edges;
  const double low_threshold = 254;
  const double high_threshhold = 255;
  Canny(img, edges, low_threshold, high_threshhold, 3);
  return edges;
}
void image_processor::save_image(Mat img, const std::string &filename) {
  if (img.empty())
    return;
  auto file_path = folder / (filename + ".png");
  imwrite(file_path.string(), img);
}
std::ofstream image_processor::make_data_file() {
  auto full_path = folder / data_file;
  std::ofstream f(full_path);
  // TODO: Move to data fill function
  f << ";Average point;Average deviation;Max deviation\n";
  return std::ofstream();
}
void image_processor::set_parameter(parameter param, std::any value) {
  auto it = parameters.find(param);
  if (it == parameters.end()) {
    parameters.emplace(param, value);
    return;
  }
  if (it->second.type() != value.type()) {
    auto message = fmt::format("Wrong parameter type. Type {} instead of {}",
                               value.type().name(), it->second.type().name());
    char *what = new char[message.length() + 1];
    // strcpy(what, message.c_str());
    // throw std::exception(what);
  }
  it->second = value;
}
Mat image_processor::cvt_non_white_to_black(Mat img) {
  Mat bw_img;
  threshold(img, bw_img, 254, 255, THRESH_BINARY);
  return bw_img;
}
void image_processor::set_original_image(Mat img) { original = img; }
void image_processor::set_save_name(std::string name) {
  save_name = std::move(name);
}
void image_processor::set_folder(std::string folder) {}
void image_processor::generate() {}
} // namespace sf
