#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace sf {
using namespace cv;

class image_processor {
protected:
  enum criterion : size_t {
    average_point,
    maximum_deviation,
    average_deviation,

    enum_count
  };
  using criterion_array = std::array<float, criterion::enum_count>;

  std::filesystem::path folder;
  std::filesystem::path data_name{"data.csv"};
  std::map<std::string_view, criterion_array> method_data;
  std::map<std::string_view, std::vector<cv::Point>> lines;

  Mat original;

  float center_x = 0.535f;
  float center_y = 0.5f;
  float shift_x = 0.05f;
  float shift_y = 0.03f;

  template <class F, class... Args>
  Mat apply_filter(Mat img, F func, Args... args) {
    Mat filtered_img;
    std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
    return filtered_img;
  }
  template <class F>
  Mat process_img(Mat img, std::string_view method_name, F func) {
    Mat f_img = std::invoke(func, img);

    save_image(f_img, folder, std::string(method_name) + "_f");

    auto line_points = find_line_points(f_img);
    Mat line_img;
    cvtColor(img, line_img, COLOR_GRAY2BGR);
    draw_line(line_img, line_points);
    save_image(crop_img(line_img), folder, std::string(method_name) + "_line");

    lines.emplace(method_name, std::move(line_points));
  }

  Mat cvt_non_white_to_black(Mat img);
  Mat detect_edges(Mat img);
  void save_image(Mat img, const std::string &name);

  std::vector<Point> find_line_points(Mat img);
  std::ofstream make_data_file();
  criterion_array form_data(const std::vector<Point> &line_points);
  void write_data();

  void draw_line(Mat img, const std::vector<Point> &points,
                 Scalar line_color = {0, 0, 255});
  Mat crop_img(Mat img);

  ~image_processor() {}

public:
  void set_original_image(Mat img);
  void set_save_name(std::filesystem::path name);
  void set_folder(std::filesystem::path folder = "");
  virtual void generate() = 0;
};
} // namespace sf
