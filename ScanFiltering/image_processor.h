#pragma once

#include "processor_parameters.h"
#include <any>
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
  static constexpr std::string_view data_file{"data.csv"};
  enum static constexpr std::string_view av_p{"average_point"};
  static constexpr std::string_view max_dev{"maximum_deviation"};
  static constexpr std::string_view av_dev{"average_deviation"};

protected:
  std::filesystem::path folder;
  std::string save_name;
  Mat original;
  Mat kernel = Mat::ones(3, 3, CV_8UC1);
  std::map<parameter, std::any> parameters;

  template <class F, class... Args>
  Mat apply_filter(Mat img, F func, Args... args) {
    Mat filtered_img;
    std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
    return filtered_img;
  }

  Mat cvt_non_white_to_black(Mat img);
  Mat detect_edges(Mat img);
  void save_image(Mat img, const std::string &name);

  std::ofstream make_data_file();
  std::map<std::string, float> form_data(const std::vector<Point> &line_points);
  void
  write_data(std::map<std::string, std::map<std::string_view, float>> data);

  ~image_processor() {}

public:
  void set_parameter(parameter param, std::any value);
  void set_original_image(Mat img);
  void set_save_name(std::string name);
  void set_folder(std::string = "");
  virtual void generate() = 0;
};
} // namespace sf
