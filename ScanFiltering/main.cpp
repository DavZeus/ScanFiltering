#include <filesystem>
#include <fmt/core.h>
#include <functional>
#include <numeric>
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
  std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
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
  // Mat kernel = Mat::ones(3, 3, img.type());
  Mat kernel = (Mat_(3, 3, CV_8SC1) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
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

Mat apply_fillter2d(Mat img, bool show = true) {
  Mat kernel = Mat::ones(3, 3, CV_32F) / 9.F;
  Point anchor(-1, -1);
  return apply_filter(img, show, &filter2D, img.type(), kernel, anchor, 0,
                      BORDER_DEFAULT);
}

Mat apply_blur(Mat img, bool show = true) {
  Size size(3, 3);
  Point anchor(-1, -1);
  return apply_filter(img, show, &blur, size, anchor, BORDER_DEFAULT);
}

Mat apply_box_filter(Mat img, bool show = true) {
  Size size(3, 3);
  Point anchor(-1, -1);
  return apply_filter(img, show, &boxFilter, -1, size, anchor, true,
                      BORDER_DEFAULT);
}

Mat apply_bilateral_filter(Mat img, bool show = true) {
  int d = 7;
  int sigma_c = 14;
  int sigma_s = 3;
  return apply_filter(img, show, &bilateralFilter, d, sigma_c, sigma_s,
                      BORDER_DEFAULT);
}

Mat apply_gaussian_blur(Mat img, bool show = true) {
  Size size(3, 3);
  return apply_filter(img, show, &GaussianBlur, size, 0, 0, BORDER_DEFAULT);
}

Mat apply_median_blur(Mat img, bool show = true) {
  int size = 3;
  return apply_filter(img, show, &medianBlur, size);
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

void avg(Mat img, std::string folder, std::string save_name) {
  std::vector<int> points(img.rows);
  for (int i = 0; i < img.rows; ++i) {
    Mat_<uchar> row = img.row(i);
    auto first = std::find(row.begin(), row.end(), 255);
    auto last = std::find(row.rbegin(), row.rend(), 255);
    if (first != row.end() && last != row.rend()) {
      points.emplace_back(
          (first.pos() + (last.base().pos() - first.pos()) / 2).x);
    }
  }
  if (!points.empty()) {
    int average =
        std::accumulate(points.begin(), points.end(), 0) / points.size();
    std::vector<int> deviation(points.size());
    for (auto p : points) {
      deviation.emplace_back(std::abs(p - average));
    }
    std::max
  }
}

template <class F>
void filter_img(Mat img, std::string folder, std::string save_name, F func,
                bool show = true) {
  Mat f_img = func(img, show);
  Mat e_f_img = detect_edges(f_img, show);
  Mat bw_f_img = cvt_non_white_to_black(f_img, show);
  Mat e_bw_f_img = detect_edges(bw_f_img, show);
  Mat otsu_f_img;
  threshold(f_img, otsu_f_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
  Mat e_otsu_f_img = detect_edges(bw_f_img, show);

  save_image(f_img, folder, save_name);
  save_image(e_f_img, folder, save_name + "_edges");
  save_image(bw_f_img, folder, save_name + "_bw");
  save_image(e_bw_f_img, folder, save_name + "_bw_edges");
  save_image(otsu_f_img, folder, save_name + "_otsu");
  save_image(e_otsu_f_img, folder, save_name + "_otsu_edges");
}

int main(int argc, char *argv[]) {
  using namespace cv;
  const bool show_images = false;

  if (argc < 2) {
    fmt::print("Not enough arguments. File path is needed\n");
    return EXIT_FAILURE;
  }

  if (show_images) {
    namedWindow(window_name.data(), WindowFlags::WINDOW_KEEPRATIO);
  }

  Mat img;
  cvtColor(imread(argv[1]), img, COLOR_RGB2GRAY);

  std::string folder = make_save_folder();
  Mat thr;
  threshold(img, thr, 0, 255, THRESH_BINARY | THRESH_OTSU);
  save_image(thr, folder, "otsu");
  save_image(detect_edges(img, show_images), folder, "orignal_edges");
  filter_img(img, folder, "closer", &apply_closer, show_images);
  filter_img(img, folder, "opening", &apply_opening, show_images);
  filter_img(img, folder, "custom", &apply_custom, show_images);
  filter_img(img, folder, "dilate", &apply_dilate, show_images);
  filter_img(img, folder, "erode", &apply_erode, show_images);
  filter_img(img, folder, "filter2d", &apply_fillter2d, show_images);
  filter_img(img, folder, "blur", &apply_blur, show_images);
  filter_img(img, folder, "box_filter", &apply_box_filter, show_images);
  filter_img(img, folder, "bilateral_filter", &apply_bilateral_filter,
             show_images);
  filter_img(img, folder, "gaussian_blur", &apply_gaussian_blur, show_images);
  filter_img(img, folder, "median_blur", &apply_median_blur, show_images);
}
