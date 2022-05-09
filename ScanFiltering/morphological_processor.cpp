#include "morphological_processor.h"

#include <opencv2/imgproc.hpp>

namespace sf {

Mat morphological_processor::apply_dilate(Mat img) {
  return apply_filter(img, dilate, kernel, anchor, iteration_number,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}
Mat morphological_processor::apply_erode(Mat img) {
  return apply_filter(img, &erode, kernel, anchor, iteration_number,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}
Mat morphological_processor::apply_closer(Mat img) {
  return apply_filter(img, &morphologyEx, MORPH_CLOSE, kernel, anchor,
                      iteration_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}
Mat morphological_processor::apply_opening(Mat img) {
  return apply_filter(img, &morphologyEx, MORPH_OPEN, kernel, anchor,
                      iteration_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}
Mat morphological_processor::apply_custom_closer(Mat img) {
  unsigned counter = custom_iteration_number;
  while (counter--) {
    Mat iteration_img = apply_dilate(img);
    img = apply_erode(iteration_img);
  }
  return img;
}
Mat morphological_processor::apply_custom_opening(Mat img) {
  unsigned counter = custom_iteration_number;
  while (counter--) {
    Mat iteration_img = apply_erode(img);
    img = apply_dilate(img);
  }
  return img;
}
void morphological_processor::set_kernel(Mat kernel) { this->kernel = kernel; }
void morphological_processor::set_anchor(Point anchor) {
  this->anchor = anchor;
}
void morphological_processor::set_iteration_number(int num) {
  iteration_number = num;
}
void morphological_processor::set_custom_iteration_number(int num) {
  custom_iteration_number = num;
}
std::map<std::string_view, Mat> morphological_processor::apply_all_filters() {
  std::map<std::string_view, Mat> f_imgs;
  f_imgs.emplace("dilate", apply_dilate(original));
  f_imgs.emplace("custom_closer", apply_custom_closer(original));
  f_imgs.emplace("closer", apply_closer(original));
  f_imgs.emplace("custom_opening", apply_custom_opening(original));
  f_imgs.emplace("erode", apply_erode(original));
  f_imgs.emplace("opening", apply_opening(original));
  return f_imgs;
}
} // namespace sf
