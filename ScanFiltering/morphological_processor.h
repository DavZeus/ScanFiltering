#pragma once
#include "image_processor.h"

namespace sf {
using namespace cv;

class morphological_processor : public image_processor {
  template <class F, class... Args>
  Mat apply_morphological_filter(Mat img, F func, Args... args) {
    Point anchor(-1, -1);
    int iteration_number = 3;
    return apply_filter(img, &morphologyEx, args..., BORDER_CONSTANT,
                        morphologyDefaultBorderValue());
  }

  template <class F, class... Args> Mat apply_custom_filter() {}

public:
  morphological_processor();
};
} // namespace sf
