#pragma once

#include <functional>
#include <map>
#include <opencv2/core/mat.hpp>

namespace sf {
using namespace cv;

class image_processor {
protected:
  Mat original;

  template <class F, class... Args>
  Mat apply_filter(Mat img, F func, Args... args) {
    Mat filtered_img;
    std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
    return filtered_img;
  }

  ~image_processor() {}

public:
  void set_original_image(Mat img);
  virtual std::map<std::string_view, Mat> apply_all_filters() = 0;
};
} // namespace sf
