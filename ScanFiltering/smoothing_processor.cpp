#include "smoothing_processor.h"

#include <opencv2/imgproc.hpp>

namespace sf {
Mat smoothing_processor::apply_blur(Mat img) {
  return apply_filter(img, &blur, ksize, anchor, BORDER_DEFAULT);
}
Mat smoothing_processor::apply_bilateral_filter(Mat img) {
  return apply_filter(img, &bilateralFilter, d, sigma_c, sigma_s,
                      BORDER_DEFAULT);
}
Mat smoothing_processor::apply_gaussian_blur(Mat img) {
  return apply_filter(img, &GaussianBlur, ksize, 0, 0, BORDER_DEFAULT);
}
Mat smoothing_processor::apply_median_blur(Mat img) {
  return apply_filter(img, &medianBlur, size);
}
void smoothing_processor::set_ksize(Size ksize) { this->ksize = ksize; }
void smoothing_processor::set_size(int size) { this->size = size; }
void smoothing_processor::set_anchor(Point point) { this->anchor = anchor; }
void smoothing_processor::set_bilateral_params(int d, int sigma_c,
                                               int sigma_s) {
  this->d = d;
  this->sigma_c = sigma_c;
  this->sigma_s = sigma_s == 0 ? sigma_c : sigma_s;
}
std::map<std::string_view, Mat> smoothing_processor::apply_all_filters() {
  std::map<std::string_view, Mat> f_imgs;
  f_imgs.emplace("blur", apply_blur(original));
  f_imgs.emplace("bilateral_filter", apply_bilateral_filter(original));
  f_imgs.emplace("gaussian_blur", apply_gaussian_blur(original));
  f_imgs.emplace("median_blur", apply_median_blur(original));
  return f_imgs;
}
} // namespace sf
