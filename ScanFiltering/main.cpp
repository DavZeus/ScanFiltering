#include <array>
#include <filesystem>
#include <fmt/core.h>
#include <numbers>
#include <opencv2/opencv.hpp>

using namespace cv;

#define PRINT_TYPE(x) fmt::print("Img type: {}", x);

const std::string_view window_name = "1";

void show_image(Mat img) {
  imshow(window_name.data(), img);
  waitKey(0);
}

template <class F, class... Args>
Mat apply_filter(Mat img, bool show, F func, Args... args) {
  Mat filtered_img;
  func(img, filtered_img, std::forward<Args>(args)...);
  if (show)
    show_image(filtered_img);
  return filtered_img;
}

Mat apply_dilate(Mat img, bool show = true) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, show, &dilate, kernel, anchor, iteration_number,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}

Mat apply_erode(Mat img, bool show = true) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, show, &erode, kernel, anchor, iteration_number,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}

Mat apply_closer(Mat img, bool show = true) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, show, &morphologyEx, MORPH_CLOSE, kernel, anchor,
                      iteration_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

Mat apply_opening(Mat img, bool show = true) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, show, &morphologyEx, MORPH_OPEN, kernel, anchor,
                      iteration_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

Mat apply_custom(Mat img, bool show = true) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteratioin_number = 1;
  Mat first_stage =
      apply_filter(img, show, &dilate, kernel, anchor, iteratioin_number,
                   BORDER_CONSTANT, morphologyDefaultBorderValue());
  Mat second_stage =
      apply_filter(first_stage, show, &erode, kernel, anchor, 2,
                   BORDER_CONSTANT, morphologyDefaultBorderValue());
  return apply_filter(second_stage, show, &dilate, kernel, anchor,
                      iteratioin_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

std::string generate_time_string() {
  std::stringstream time_parse;
  tm tm;
  auto time =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  localtime_s(&tm, &time);
  time_parse << std::put_time(&tm, "%FT%H-%M-%S") << std::flush;
  return time_parse.str();
}

std::string make_save_folder(std::string path = "") {
  if (path.empty()) {
    path = generate_time_string();
  } else {
    path += "-" + generate_time_string();
  }
  std::filesystem::create_directory(path);
  return path;
}

void save_image(Mat img, std::string path, std::string filename) {
  std::filesystem::path fullname{path};
  fullname /= filename + ".png";
  imwrite(fullname.string(), img);
}

Mat detect_edges(Mat img, bool show = true) {
  Mat edges;
  const double low_threshold = 254;
  const double high_threshhold = 255;
  Canny(img, edges, low_threshold, high_threshhold, 3);
  if (show)
    show_image(edges);
  return edges;
}

Mat cvt_non_white_to_black(Mat img, bool show = true) {
  Mat bw_img = img.clone();
  bw_img.forEach<uint8_t>([](uint8_t &p, const int *pos) {
    if (p != 255)
      p = 0;
  });
  if (show)
    show_image(bw_img);
  return bw_img;
}

template <class F>
void filter_img(Mat img, std::string folder, std::string name, F func,
                bool show = true) {
  Mat f_img = func(img, show);
  Mat e_f_img = detect_edges(f_img, show);
  Mat bw_f_img = cvt_non_white_to_black(f_img, show);
  Mat e_bw_f_img = detect_edges(bw_f_img, show);

  save_image(f_img, folder, name);
  save_image(e_f_img, folder, name + "-edges");
  save_image(bw_f_img, folder, name + "-bw");
  save_image(e_bw_f_img, folder, name + "-bw-edges");
}

int main() {
  using namespace cv;
  const bool show_images = false;

  if (show_images) {
    namedWindow(window_name.data(), WindowFlags::WINDOW_KEEPRATIO);
  }

  Mat img = imread("20k_edge.bmp");
  Mat gray_img;
  cvtColor(img, gray_img, COLOR_RGB2GRAY);

  std::string folder = make_save_folder();
  save_image(detect_edges(gray_img, show_images), folder, "orignal-edges");
  filter_img(gray_img, folder, "cl", &apply_closer, show_images);
  filter_img(gray_img, folder, "op", &apply_opening, show_images);
  filter_img(gray_img, folder, "custom", &apply_custom, show_images);
  // std::array filtered_images = {
  //  apply_filter(gray_img, true, &filter2D, gray_img.type(), Mat::ones(3,
  //  3, CV_32F) / static_cast<float>(9), Point(-1, -1), 0, BORDER_DEFAULT),
  //  apply_filter(gray_img, true, &blur, Size(3, 3), Point(-1, -1),
  //  BORDER_DEFAULT),
  //  apply_filter(gray_img, true, &boxFilter, gray_img.type(), Size(3, 3),
  //  Point(-1, -1), true, BORDER_DEFAULT),
  //  apply_filter(gray_img, true, &bilateralFilter, 9, 50, 50,
  //  BORDER_DEFAULT),
  //  apply_filter(gray_img, true, &GaussianBlur, cv::Size(3, 3), 0, 0,
  //  BORDER_DEFAULT),
  //  apply_filter(gray_img, true, &medianBlur, 3),
  //  apply_closer(gray_img),
  //  apply_opening(gray_img),
  //};
}
