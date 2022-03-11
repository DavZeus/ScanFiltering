#pragma once
#include "image_processor.h"

namespace sf {
using namespace cv;

class morphological_processor : public image_processor {
public:
  enum parameter_name { anchor, iteration_number, custom_iteration_number };
  morphological_processor() {
    parameters.emplace(anchor, Point{-1, -1});
    parameters.emplace(iteration_number, 3);
    parameters.emplace(custom_iteration_number, 1);
  }

private:
  template <class F, class... Args>
  Mat apply_morphological_filter(Mat img, F func, Args... args) {
    Point anchor(-1, -1);
    int iteration_number = 3;
    return apply_filter(img, &morphologyEx, MORPH_CLOSE, kernel, anchor,
                        iteration_number, BORDER_CONSTANT,
                        morphologyDefaultBorderValue());
  }

  template <class F, class... Args> Mat apply_custom_filter() {}
};
} // namespace sf
