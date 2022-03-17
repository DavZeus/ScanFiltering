#include "image_processor.h"

#include <execution>
#include <fmt/format.h>
#include <numeric>

#define _STR(x) "_" #x
#define STR(x) #x

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
}
std::map<std::string, float>
image_processor::form_data(const std::vector<Point> &line_points) {
  const float average_point =
      std::reduce(std::execution::par, line_points.begin(), line_points.end())
          .y /
      static_cast<float>(line_points.size());
  std::vector<float> deviation;
  deviation.reserve(line_points.size());
  for (const auto &p : line_points) {
    deviation.emplace_back(std::abs(p.y - average_point));
  }
  const float max_deviation =
      *std::max_element(deviation.begin(), deviation.end());
  const float average_deviation =
      std::reduce(deviation.begin(), deviation.end()) / deviation.size();
  std::map<std::string, float> data;
  data.emplace(av_p, average_point);
  data.emplace(max_dev, max_deviation);
  data.emplace(av_dev, average_deviation);
  return data;
}
void image_processor::write_data(
    std::map<std::string, std::map<std::string, float>> data) {
  auto file = make_data_file();
  if (!file) {
    fmt::print("Can not open data file\n");
    return;
  }
  std::string r =
      fmt::format(std::locale("ru_RU.UTF-8"), "{};{:.2Lf};{:.2Lf};{:.2Lf}\n",
                  name, data.at("average_point"), data.at("average_deviation"),
                  data.at("max_deviation"));
  f << std::regex_replace(r, std::regex{"Â"}, "");
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
    strcpy(what, message.c_str());
    throw std::exception(what);
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
void image_processor::set_folder(std::string) {}
void image_processor::generate() {}
} // namespace sf
