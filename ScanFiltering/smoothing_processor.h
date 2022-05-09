#pragma once
#include "image_processor.h"

namespace sf {
using namespace cv;

class smoothing_processor : public image_processor {
  Size ksize{3, 3};
  int size = 3;
  Point anchor{-1, -1};
  int d = 9;
  int sigma_c = 5;
  int sigma_s = 5;

public:
  Mat apply_blur(Mat img);
  Mat apply_bilateral_filter(Mat img);
  Mat apply_gaussian_blur(Mat img);
  Mat apply_median_blur(Mat img);

  void set_ksize(Size ksize);
  void set_size(int size);
  void set_anchor(Point point);
  void set_bilateral_params(int d, int sigma_c, int sigma_s = 0);

  std::map<std::string_view, Mat> apply_all_filters();
};
} // namespace sf
