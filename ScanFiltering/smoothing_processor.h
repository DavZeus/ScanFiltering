#pragma once
#include "image_processor.h"

namespace sf {
using namespace cv;

class smoothing_processor : public image_processor {
public:
  smoothing_processor();
};
} // namespace sf
