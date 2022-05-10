#pragma once

#include <filesystem>
#include <map>
#include <opencv2/core/mat.hpp>
#include <string>

namespace sf {
using namespace cv;
class analyzer {
  inline static constexpr size_t relative_shift = 1;
  enum criterion : size_t {
    avg_point,
    avg_deviation,
    avg_deviation_relative = avg_deviation + relative_shift,
    avg_deviation_percent,
    max_deviation,
    max_deviation_relative = max_deviation + relative_shift,
    max_deviation_percent,

    enum_count
  };
  using criterion_array = std::array<float, enum_count>;

  using line = std::vector<Point>;

  using filter_data = std::pair<const std::string_view, criterion_array>;
  using map_of_criterion_data = std::map<std::string_view, criterion_array>;
  using map_of_noise_distribution =
      std::map<std::string_view, std::pair<size_t, size_t>>;
  using map_of_lines = std::map<std::string_view, line>;

  std::filesystem::path folder;
  std::filesystem::path common_name;
  std::filesystem::path data_filename{"data.csv"};
  std::filesystem::path distribution_filename{"distrib.csv"};

  line find_line_points(Mat img);
  map_of_lines form_all_line_points(std::map<std::string_view, Mat> imgs);

  float find_avg_point(const line &line);
  std::vector<float> find_deviation(const line &line, float avg_point);
  float find_max_deviation(const std::vector<float> deviation);
  float find_avg_deviation(const std::vector<float> deviation);
  float find_percent(float average, float deviation);
  criterion_array form_line_independent_data(const line &line);
  map_of_criterion_data form_all_independent_data(const map_of_lines &lines);

  float find_min_criterion_value(const map_of_criterion_data &data,
                                 criterion name);
  void form_relative_data_by_criterion(map_of_criterion_data &data,
                                       criterion name);
  void form_all_relative_data(map_of_criterion_data &data);

  std::map<unsigned, unsigned> form_distribution(const line &points,
                                                 int avg_point);
  void write_distribution_data(
      std::map<std::string_view, std::map<unsigned, unsigned>> data);
  void make_deviation_distribution(const map_of_criterion_data &data,
                                   const map_of_lines &lines);

  std::ofstream make_data_file(const std::filesystem::path &filename);

  std::string form_data_header();
  std::string form_line_values(const criterion_array &values);
  void write_data(map_of_criterion_data data);

  Mat cvt_non_white_to_black(Mat img);

public:
  map_of_lines generate_data(std::map<std::string_view, Mat> imgs);

  void set_folder(std::filesystem::path folder);
  void set_common_filename(std::filesystem::path filename);
  void set_data_filename(std::filesystem::path filename);
  void set_distribution_filename(std::filesystem::path filename);
};
} // namespace sf
