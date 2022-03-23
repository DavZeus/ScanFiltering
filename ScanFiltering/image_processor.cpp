#include "image_processor.h"

#include <execution>
#include <fmt/format.h>
#include <numeric>
#include <regex>

#include "custom_utility.h"
#include "io_operations.h"

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
std::vector<Point> image_processor::find_line_points(Mat img) {
  std::vector<Point> line_points;
  line_points.reserve(img.cols);
  for (int i = 0; i < img.cols; ++i) {
    Mat_<uchar> col = img.col(i);
    auto first = std::find(col.begin(), col.end(), 255);
    if (first == col.end())
      continue;
    auto last = std::find(col.rbegin(), col.rend(), 255);
    line_points.emplace_back(i, ((last.base().pos() + first.pos()) / 2).y);
  }
  line_points.shrink_to_fit();
  return line_points;
}
std::ofstream image_processor::make_data_file() {
  auto full_path = folder / data_name;
  std::ofstream f(full_path);
}
std::array<float, image_processor::criterion::enum_count>
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
  criterion_array data{};
  data.at(criterion::average_point) = average_point;
  data.at(criterion::maximum_deviation) = max_deviation;
  data.at(criterion::average_deviation) = average_deviation;
  return data;
}
void image_processor::write_data() {
  auto file = make_data_file();
  if (!file) {
    fmt::print("Can not create/open data file\n");
    return;
  }
  file << ";Average point;Average deviation;Max deviation\n";
  for (const auto &[method_name, method_data] : method_data) {
    std::string line = fmt::format(
        std::locale("ru_RU.UTF-8"), "{};{:.2Lf};{:.2Lf};{:.2Lf}\n", method_name,
        method_data.at(cu::tot(criterion::average_point)),
        method_data.at(cu::tot(criterion::average_deviation)),
        method_data.at(cu::tot(criterion::maximum_deviation)));
    cu::remove_ru_separator(line);
    file << line;
  }
}
void image_processor::draw_line(Mat img, const std::vector<Point> &points,
                                Scalar line_color) {
  if (img.type() != CV_8UC3 || points.empty())
    return;
  for (auto p1 = points.begin(), p2 = points.begin() + 1; p2 != points.end();
       ++p1, ++p2) {
    line(img, *p1, *p2, line_color);
  }
}
Mat image_processor::crop_img(Mat img) {
  const float first_x = center_x - shift_x;
  const float second_x = center_x + shift_x;
  const float first_y = center_y - shift_y;
  const float second_y = center_y + shift_y;
  const int x0 = static_cast<int>(img.rows * first_y);
  const int x1 = static_cast<int>(img.rows * second_y);
  const int y0 = static_cast<int>(img.cols * first_x);
  const int y1 = static_cast<int>(img.cols * second_x);
  return img(Range(x0, x1), Range(y0, y1));
}
Mat image_processor::cvt_non_white_to_black(Mat img) {
  Mat bw_img;
  threshold(img, bw_img, 254, 255, THRESH_BINARY);
  return bw_img;
}
void image_processor::set_original_image(Mat img) { original = img; }
void image_processor::set_save_name(std::filesystem::path name) {
  if (name.empty()) {
    return;
  }
  if (name.extension() != ".csv") {
    name.append(".csv");
  }
  data_name = std::move(name);
}
void image_processor::set_folder(std::filesystem::path folder) {
  if (folder.empty()) {
    this->folder = io::make_save_folder();
  } else {
    this->folder = std::move(folder);
  }
}
} // namespace sf
