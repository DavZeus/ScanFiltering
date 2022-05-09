#pragma once
#include "image_processor.h"

namespace sf {
using namespace cv;

class morphological_processor : public image_processor {
  Mat kernel = Mat::ones(3, 3, CV_8UC1);
  Point anchor = Point{-1, -1};
  int iteration_number = 3;
  int custom_iteration_number = 1;

public:
  Mat apply_dilate(Mat img);
  Mat apply_erode(Mat img);
  Mat apply_closer(Mat img);
  Mat apply_opening(Mat img);
  Mat apply_custom_closer(Mat img);
  Mat apply_custom_opening(Mat img);

  // TODO: Add checks
  void set_kernel(Mat kernel);
  void set_anchor(Point anchor);
  void set_iteration_number(int num);
  void set_custom_iteration_number(int num);

  std::map<std::string_view, Mat> apply_all_filters();
};
} // namespace sf
