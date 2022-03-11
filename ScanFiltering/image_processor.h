#pragma once

#include <filesystem>
#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace sf {
using namespace cv;

class image_processor {
  std::filesystem::path folder;
  Mat original;

  template <class F, class... Args>
  Mat apply_filter(Mat img, F func, Args... args) {
    Mat filtered_img;
    std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
    return filtered_img;
  }

  template <class F, class... Args>
  Mat apply_morphological_filter(Mat img, F func, Args... args) {
    Point anchor(-1, -1);
    int iteration_number = 3;
    return apply_filter(img, &morphologyEx, MORPH_CLOSE, kernel, anchor,
                        iteration_number, BORDER_CONSTANT,
                        morphologyDefaultBorderValue());
  };
} // namespace sf
