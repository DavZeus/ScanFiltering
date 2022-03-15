#include "image_processor.h"

namespace sf {
bool image_processor::set_parameter(parameter param, std::any value) {
  auto it = parameters.find(param);
  if (it == parameters.end()) {
    parameters.emplace(param, value);
    return true;
  }
  if (it->second.type() != value.type())
    return false;
  it->second = value;
  return true;
}
Mat image_processor::cvt_non_white_to_black(Mat img) {
  Mat bw_img;
  threshold(img, bw_img, 254, 255, THRESH_BINARY);
  return bw_img;
}
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
void image_processor::set_original(Mat img) { original = img; }
} // namespace sf
