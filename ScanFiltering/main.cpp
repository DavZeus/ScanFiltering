#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#define _STR(x) "_" #x
#define STR(x) #x
using namespace cv;

const std::string_view window_name = "1";
const std::string_view data_filename = "big_data.csv";

template <class F, class... Args>
Mat apply_filter(Mat img, F func, Args... args) {
  Mat filtered_img;
  std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
  return filtered_img;
}

Mat apply_dilate(Mat img) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, dilate, kernel, anchor, iteration_number,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}

Mat apply_erode(Mat img) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, &erode, kernel, anchor, iteration_number,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}

Mat apply_closer(Mat img) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, &morphologyEx, MORPH_CLOSE, kernel, anchor,
                      iteration_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

Mat apply_opening(Mat img) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, &morphologyEx, MORPH_OPEN, kernel, anchor,
                      iteration_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

Mat apply_custom_closer(Mat img) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteratioin_number = 1;
  int custom_iteration = 3;
  while (custom_iteration--) {
    Mat iteration_img =
        apply_filter(img, &dilate, kernel, anchor, iteratioin_number,
                     BORDER_CONSTANT, morphologyDefaultBorderValue());
    img = apply_filter(iteration_img, &erode, kernel, anchor, iteratioin_number,
                       BORDER_CONSTANT, morphologyDefaultBorderValue());
  }
  return img;
}

Mat apply_custom_opening(Mat img) {
  Mat kernel = Mat::ones(3, 3, img.type());
  Point anchor(-1, -1);
  int iteratioin_number = 1;
  int custom_iteration = 3;
  while (custom_iteration--) {
    Mat iteration_img =
        apply_filter(img, &erode, kernel, anchor, iteratioin_number,
                     BORDER_CONSTANT, morphologyDefaultBorderValue());
    img =
        apply_filter(iteration_img, &dilate, kernel, anchor, iteratioin_number,
                     BORDER_CONSTANT, morphologyDefaultBorderValue());
  }
  return img;
}

Mat apply_blur(Mat img) {
  Size size(3, 3);
  Point anchor(-1, -1);
  return apply_filter(img, &blur, size, anchor, BORDER_DEFAULT);
}

Mat apply_bilateral_filter(Mat img) {
  int d = 7;
  int sigma_c = 14;
  int sigma_s = 3;
  return apply_filter(img, &bilateralFilter, d, sigma_c, sigma_s,
                      BORDER_DEFAULT);
}

Mat apply_gaussian_blur(Mat img) {
  Size size(3, 3);
  return apply_filter(img, &GaussianBlur, size, 0, 0, BORDER_DEFAULT);
}

Mat apply_median_blur(Mat img) {
  int size = 3;
  return apply_filter(img, &medianBlur, size);
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

void save_image(Mat img, const std::string &path, const std::string &filename) {
  std::filesystem::path fullname{path};
  fullname /= filename + ".png";
  imwrite(fullname.string(), img);
}

Mat detect_edges(Mat img) {
  Mat edges;
  const double low_threshold = 254;
  const double high_threshhold = 255;
  Canny(img, edges, low_threshold, high_threshhold, 3);
  return edges;
}

Mat cvt_to_bw(Mat img) {
  Mat bw_img = img.clone();
  bw_img.forEach<uint8_t>([](uint8_t &p, const int *pos) {
    if (p != 255)
      p = 0;
  });
  return bw_img;
}

auto find_line_points(Mat img) {
  std::vector<Point> line_points;
  line_points.reserve(img.cols);
  for (int i = 0; i < img.cols; ++i) {
    Mat_<uchar> col = img.col(i);
    auto first = std::find(col.begin(), col.end(), 255);
    if (first == col.end())
      continue;
    auto last = std::find(col.rbegin(), col.rend(), 255);
    line_points.emplace_back(i, ((last.base().pos() + first.pos()) / 2).y);
  }
  line_points.shrink_to_fit();
  return line_points;
}

void draw_line(Mat img, const std::string &folder, const std::string &save_name,
               const std::vector<Point> &points) {
  if (points.empty()) {
    return;
  }
  Mat img_with_line;
  cvtColor(img, img_with_line, COLOR_GRAY2BGR);
  for (auto p1 = points.begin(), p2 = points.begin() + 1; p2 != points.end();
       ++p1, ++p2) {
    line(img_with_line, *p1, *p2, {0, 0, 255});
    line(img_with_line, *p1, *p1, {255, 0, 0});
  }
  save_image(img_with_line, folder, save_name + "_line");
}

auto form_data(const std::vector<Point> &line_points) {
  const float average_point =
      std::reduce(line_points.begin(), line_points.end()).y /
      static_cast<float>(line_points.size());
  std::vector<float> deviation;
  deviation.reserve(line_points.size());
  for (auto p : line_points) {
    deviation.emplace_back(std::abs(p.y - average_point));
  }
  const float max_deviation =
      *std::max_element(deviation.begin(), deviation.end());
  const float average_deviation =
      std::reduce(deviation.begin(), deviation.end()) / deviation.size();
  std::map<std::string, float> data;
  data.emplace(STR(average_point), average_point);
  data.emplace(STR(max_deviation), max_deviation);
  data.emplace(STR(average_deviation), average_deviation);
  return data;
}

void write_data(const std::string &folder, const std::string &name,
                std::map<std::string, float> data) {
  std::filesystem::path full_path(folder);
  full_path /= data_filename;
  std::ofstream f(full_path, std::ios::app);
  if (!f) {
    fmt::print("Can not open data file\n");
    return;
  }
  f.imbue(std::locale("ru_RU.UTF-8"));
  f << std::fixed << std::setprecision(2) << name << ';'
    << data.at("average_point") << ';' << data.at("average_deviation") << ';'
    << data.at("max_deviation") << '\n';
}

auto make_data(Mat img, const std::string &folder,
               const std::string &save_name) {
  auto line_points = find_line_points(img);
  write_data(folder, save_name, form_data(line_points));
  return line_points;
}

template <class F>
void filter_img(Mat img, std::string folder, std::string save_name, F func) {
  Mat f_img = std::invoke(func, img);
  Mat e_f_img = detect_edges(f_img);
  Mat bw_f_img = cvt_to_bw(f_img);
  Mat e_bw_f_img = detect_edges(bw_f_img);

  save_image(f_img, folder, save_name + _STR(f_img));
  save_image(e_f_img, folder, save_name + _STR(e_f_img));
  save_image(bw_f_img, folder, save_name + _STR(bw_f_img));
  save_image(e_bw_f_img, folder, save_name + _STR(e_bw_f_img));

  const auto line_points =
      make_data(bw_f_img, folder, save_name + _STR(bw_f_img));
  draw_line(img, folder, save_name, line_points);
}

void make_csv(const std::string &folder) {
  std::filesystem::path fullpath(folder);
  fullpath /= data_filename.data();
  std::ofstream f(fullpath);
  f << ";Average point;Max deviation;Average deviation\n";
}

void add_original_data(Mat img, const std::string &folder) {
  Mat bw_img = cvt_to_bw(img);
  const std::string name("original_bw");
  save_image(bw_img, folder, name);
  const auto line_points = make_data(bw_img, folder, name);
  draw_line(img, folder, "original", line_points);
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    fmt::print("Not enough arguments. File path is needed\n");
    return EXIT_FAILURE;
  }

  Mat img;
  cvtColor(imread(argv[1]), img, COLOR_RGB2GRAY);

  std::string folder = make_save_folder();
  make_csv(folder);
  add_original_data(img, folder);
  save_image(detect_edges(img), folder, "orignal_edges");
  filter_img(img, folder, "closer", &apply_closer);
  filter_img(img, folder, "opening", &apply_opening);
  filter_img(img, folder, "custom_closer", &apply_custom_closer);
  filter_img(img, folder, "custom_opening", &apply_custom_opening);
  filter_img(img, folder, "dilate", &apply_dilate);
  filter_img(img, folder, "erode", &apply_erode);
  filter_img(img, folder, "blur", &apply_blur);
  filter_img(img, folder, "bilateral_filter", &apply_bilateral_filter);
  filter_img(img, folder, "gaussian_blur", &apply_gaussian_blur);
  filter_img(img, folder, "median_blur", &apply_median_blur);
}
