#pragma once
#include "image_processor.h"

namespace sf {
using namespace cv;
Mat kernel = Mat::ones(3, 3, CV_8UC1);
Point anchor = Point{-1, -1};
int iteration_number = 3;
int custom_iteration_number = 1;

class morphological_processor : public image_processor {
  template <class F> Mat apply_morphological_filter(Mat img, F func) {
    return apply_filter(img, &morphologyEx, kernel, anchor, iteration_number,
                        BORDER_CONSTANT, morphologyDefaultBorderValue());
  }
  template <class F, class... Args> Mat apply_custom_filter() {}

  Mat process_dilate(Mat img);

public:
  void generate();
};
} // namespace sf
