#include "analyzer.h"

#include <execution>
#include <fmt/format.h>
#include <fstream>
#include <opencv2/imgproc.hpp>

#include "custom_utility.h"
#include "io_operations.h"

#define STR(x) #x

namespace sf {
using namespace cv;
analyzer::line analyzer::find_line_points(Mat img) {
  line points;
  points.reserve(img.cols);
  for (int i = 0; i < img.cols; ++i) {
    Mat_<uchar> col = img.col(i);
    auto first = std::find(col.begin(), col.end(), 255);
    if (first == col.end())
      continue;
    auto last = std::find(col.rbegin(), col.rend(), 255);
    points.emplace_back(i, ((last.base().pos() + first.pos()) / 2).y);
  }
  points.shrink_to_fit();
  return points;
}
analyzer::map_of_lines
analyzer::form_all_line_points(std::map<std::string_view, Mat> imgs) {
  map_of_lines lines;
  for (const auto &[name, img] : imgs) {
    lines.emplace(name, find_line_points(img));
  }
  return lines;
}
float analyzer::find_avg_point(const line &points) {
  return std::reduce(std::execution::par, points.begin(), points.end()).y /
         static_cast<float>(points.size());
}
std::vector<float> analyzer::find_deviation(const line &points,
                                            float avg_point) {
  std::vector<float> deviation;
  deviation.reserve(points.size());
  for (const auto &p : points) {
    deviation.emplace_back(std::abs(p.y - avg_point));
  }
  return deviation;
}
float analyzer::find_max_deviation(const std::vector<float> deviation) {
  return *std::max_element(deviation.begin(), deviation.end());
}
float analyzer::find_avg_deviation(const std::vector<float> deviation) {
  return std::reduce(deviation.begin(), deviation.end()) / deviation.size();
}
analyzer::criterion_array
analyzer::form_line_independent_data(const line &points) {
  const float average_point = find_avg_point(points);
  std::vector<float> deviation = find_deviation(points, average_point);
  const float maximum_deviation = find_max_deviation(deviation);
  const float average_deviation = find_avg_deviation(deviation);
  criterion_array data{};
  data.at(avg_point) = average_point;
  data.at(avg_deviation) = average_deviation;
  data.at(avg_deviation_percent) =
      std::abs(average_deviation / average_point - 1.f) * 100.f;
  data.at(max_deviation) = maximum_deviation;
  data.at(max_deviation_percent) =
      std::abs(maximum_deviation / average_point - 1.f) * 100.f;
  return data;
}

analyzer::map_of_criterion_data
analyzer::form_all_independent_data(const map_of_lines &lines) {
  map_of_criterion_data data;
  for (const auto &[filter_name, filter_line] : lines) {
    data.emplace(filter_name, form_line_independent_data(filter_line));
  }
  return data;
}

float analyzer::find_min_criterion_value(const map_of_criterion_data &data,
                                         criterion name) {
  return std::min_element(
             data.begin(), data.end(),
             [&](const filter_data &left, const filter_data &right) {
               return left.second.at(name) < right.second.at(name);
             })
      ->second.at(name);
}
void analyzer::form_relative_data_by_criterion(map_of_criterion_data &data,
                                               criterion name) {
  const float min_value = find_min_criterion_value(data, name);
  for (auto &[filter, values] : data) {
    values.at(name + relative_shift) =
        std::abs(values.at(name) / min_value - 1.f) * 100.f;
  }
}
void analyzer::form_all_relative_data(map_of_criterion_data &data) {
  form_relative_data_by_criterion(data, avg_deviation);
  form_relative_data_by_criterion(data, max_deviation);
}

std::map<unsigned, unsigned> analyzer::form_distribution(const line &points,

                                                         int avg_point) {
  std::map<unsigned, unsigned> distribution;
  for (const auto &point : points) {
    ++distribution[std::abs(point.y - avg_point)];
  }
  return distribution;
}
void analyzer::write_distribution_data(
    std::map<std::string_view, std::map<unsigned, unsigned>> data) {
  auto file = make_data_file(distribution_filename);
  if (!file) {
    fmt::print("Can not create/open distribution file\n");
    return;
  }
  for (const auto &[name, distribution] : data) {
    file << name << 'n';
    for (const auto [deviation, occurences] : distribution) {
      file << deviation << ';' << occurences << '\n';
    }
  }
}

void analyzer::make_deviation_distribution(const map_of_criterion_data &data,
                                           const map_of_lines &lines) {
  std::map<std::string_view, std::map<unsigned, unsigned>>
      distribution_of_filters;
  for (const auto &[filter_name, filter_line] : lines) {
    std::map<unsigned, unsigned> deviation_distribution;
    const int point =
        static_cast<int>(std::round(data.at(filter_name).at(avg_point)));
    distribution_of_filters.emplace(filter_name,
                                    form_distribution(filter_line, point));
  }
  write_distribution_data(std::move(distribution_of_filters));
}

std::ofstream analyzer::make_data_file(const std::filesystem::path &filename) {
  const auto full_filename = folder / filename;
  return std::ofstream(full_filename);
}
std::string analyzer::form_data_header() {
  return fmt::format(";{};{};{};{};{};{};{}\n", STR(avg_point),
                     STR(avg_deviation), STR(avg_deviation_relative),
                     STR(avg_deviation_percent), STR(max_deviation),
                     STR(max_deviation_relative), STR(max_deviation_percent));
}
std::string analyzer::form_line_values(const criterion_array &values) {
  std::string line{""};
  constexpr std::string_view patern{"{:.2Lf};"};
  const std::locale ru_loc("ru_RU.UTF-8");
  for (size_t i = 0; i < cu::tot(enum_count); i++) {
    line += fmt::format(ru_loc, patern, values.at(i));
  }
  line += '\n';
  cu::remove_wrong_ru_separator(line);
  return line;
}
void analyzer::write_data(map_of_criterion_data data) {
  auto file = make_data_file(data_filename);
  if (!file) {
    fmt::print("Can not create/open data file\n");
    return;
  }
  file << form_data_header();
  for (const auto &[filter_name, values] : data) {
    file << filter_name << ';' << form_line_values(values);
  }
}

Mat analyzer::cvt_non_white_to_black(Mat img) {
  Mat bw_img;
  threshold(img, bw_img, 254, 255, THRESH_BINARY);
  return bw_img;
}

analyzer::map_of_lines
analyzer::generate_data(std::map<std::string_view, Mat> imgs) {
  map_of_lines lines = form_all_line_points(imgs);
  map_of_criterion_data data = form_all_independent_data(lines);
  form_all_relative_data(data);
  write_data(data);
  make_deviation_distribution(data, lines);
  return lines;
}
void analyzer::set_folder(std::string folder) {
  this->folder = std::move(folder);
}
void analyzer::set_data_filename(std::string filename) {
  data_filename = std::move(filename);
}
void analyzer::set_distribution_filename(std::string filename) {
  distribution_filename = std::move(filename);
}
} // namespace sf
