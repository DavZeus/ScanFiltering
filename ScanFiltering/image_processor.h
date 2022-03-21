#pragma once

#include <any>
#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "processor_parameters.h"

namespace sf {
using namespace cv;

class image_processor {
protected:
  static constexpr std::string_view data_file{"data.csv"};

  enum class criterion {
    average_point,
    maximum_deviation,
    average_deviation,
    count
  };

  std::filesystem::path folder;
  std::string save_name;
  Mat original;
  std::map<parameter, std::any> parameter_values;
  std::map<std::string_view, std::map<criterion, float>> criterion_data;

  template <class F, class... Args>
  Mat apply_filter(Mat img, F func, Args... args) {
    Mat filtered_img;
    std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
    return filtered_img;
  }

  Mat cvt_non_white_to_black(Mat img);
  Mat detect_edges(Mat img);
  void save_image(Mat img, const std::string &name);

  std::vector<Point> find_line_points(Mat img);

  std::ofstream make_data_file();
  std::map<criterion, float> form_data(const std::vector<Point> &line_points);
  void write_data();

  void draw_line(Mat img, const std::vector<Point> &points,
                 Scalar line_color = {0, 0, 255});
  Mat crop_img(Mat img);

  ~image_processor() {}

public:
  image_processor();

  void set_parameter(parameter param, std::any value);
  void set_original_image(Mat img);
  void set_save_name(std::string name);
  void set_folder(std::string = "");
  virtual void generate() = 0;
};
} // namespace sf
