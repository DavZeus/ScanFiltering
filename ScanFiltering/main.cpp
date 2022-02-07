#include <array>
#include <filesystem>
#include <fmt/core.h>
#include <numbers>
#include <opencv2/opencv.hpp>

using namespace cv;

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
  return apply_filter(img, show, &dilate, Mat::ones(3, 3, img.type()),
                      Point(-1, -1), 3, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

Mat apply_erode(Mat img, bool show = true) {
  return apply_filter(img, show, &erode, Mat::ones(3, 3, img.type()),
                      Point(-1, -1), 3, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

Mat apply_closer(Mat img, bool show = true) {
  return apply_filter(img, show, &morphologyEx, MORPH_CLOSE,
                      Mat::ones(3, 3, img.type()), Point(-1, -1), 1,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}

Mat apply_opening(Mat img, bool show = true) {
  return apply_filter(img, show, &morphologyEx, MORPH_OPEN,
                      Mat::ones(3, 3, img.type()), Point(-1, -1), 2,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
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
  const double low_threshold = 240;
  const double high_threshhold = 250;
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

int main() {
  using namespace cv;
  namedWindow(window_name.data(), WindowFlags::WINDOW_KEEPRATIO);

  Mat img = imread("20k_edge.bmp");
  show_image(img);

  Mat gray_img;
  cvtColor(img, gray_img, COLOR_RGB2GRAY);
  detect_edges(gray_img);

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
  // apply_closer(gray_img),
  // apply_opening(gray_img),
  //};

  Mat cl_img = apply_closer(gray_img);

  std::string folder = make_save_folder();

  Mat cl_edges = detect_edges(cl_img);
  save_image(cl_edges, folder, "closer-edges");
  Mat bw_cl_img = cvt_non_white_to_black(cl_img);
  save_image(bw_cl_img, folder, "bw-closer");
  Mat bw_cl_edges = detect_edges(bw_cl_img);
  save_image(bw_cl_edges, folder, "bw-cl-edges");
  // for (const auto img : filtered_images) {
  // detect_edges(img);
  // const auto bw_img = cvt_non_white_to_black(img);
  // detect_edges(bw_img);
  //}
}
