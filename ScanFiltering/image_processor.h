#pragma once

#include <any>
#include <filesystem>
#include <functional>
#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace sf {
using namespace cv;

class image_processor {
  enum parameter_name;

protected:
  std::filesystem::path folder;
  Mat original;
  Mat kernel = Mat::ones(3, 3, CV_8UC1);
  std::map<parameter_name, std::any> parameters;
  void set_parameter(parameter_name param, std::any value) {
    auto it = parameters.find(param);
    if (it == parameters.end()) {
      parameters.emplace(param, value);
      return;
    }
    if (it->second.type() != value.type())
      return;
    it->second = value;
  }

  template <class F, class... Args>
  Mat apply_filter(Mat img, F func, Args... args) {
    Mat filtered_img;
    std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
    return filtered_img;
  }
};
} // namespace sf
