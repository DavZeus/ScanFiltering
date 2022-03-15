#pragma once

#include "processor_parameters.h"
#include <any>
#include <filesystem>
#include <functional>
#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace sf {
using namespace cv;

class image_processor {

protected:
  std::filesystem::path folder;
  Mat original;
  Mat kernel = Mat::ones(3, 3, CV_8UC1);
  std::map<parameter, std::any> parameters;
  bool set_parameter(parameter param, std::any value);

  template <class F, class... Args>
  Mat apply_filter(Mat img, F func, Args... args) {
    Mat filtered_img;
    std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
    return filtered_img;
  }

  Mat cvt_non_white_to_black(Mat img);
  Mat detect_edges(Mat img);
  void save_image(Mat img, const std::string &name);

  ~image_processor() {}

public:
  void set_original(Mat img);
  void apply();
};
} // namespace sf
