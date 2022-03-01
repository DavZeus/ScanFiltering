#include <filesystem>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <regex>
#include <vector>

#define _STR(x) "_" #x
#define STR(x) #x
using namespace cv;

const std::string_view window_name = "1";
const std::string_view data_filename = "big_data.csv";

// Cross
// Mat kernel = (Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
// Cross
// Mat kernel = (Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
// 1,
//              0, 0, 1, 0, 0, 0, 0, 1, 0, 0);
// Elipse
// Mat kernel = (Mat_<uchar>(5, 5) << 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
// 1,
//              1, 1, 1, 1, 1, 0, 1, 1, 1, 0);
// Square
Mat kernel = Mat::ones(3, 3, CV_8UC1);

template <class F, class... Args>
Mat apply_filter(Mat img, F func, Args... args) {
  Mat filtered_img;
  std::invoke(func, img, filtered_img, std::forward<Args>(args)...);
  return filtered_img;
}

Mat apply_dilate(Mat img) {
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, dilate, kernel, anchor, iteration_number,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}

Mat apply_erode(Mat img) {
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, &erode, kernel, anchor, iteration_number,
                      BORDER_CONSTANT, morphologyDefaultBorderValue());
}

Mat apply_closer(Mat img) {
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, &morphologyEx, MORPH_CLOSE, kernel, anchor,
                      iteration_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

Mat apply_opening(Mat img) {
  Point anchor(-1, -1);
  int iteration_number = 3;
  return apply_filter(img, &morphologyEx, MORPH_OPEN, kernel, anchor,
                      iteration_number, BORDER_CONSTANT,
                      morphologyDefaultBorderValue());
}

Mat apply_custom_closer(Mat img) {
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
  if (img.empty())
    return;
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

void draw_line(Mat img, const std::vector<Point> &points,
               Scalar line_colour = {0, 0, 255}) {
  if (img.type() != CV_8UC3 || points.empty())
    return;
  for (auto p1 = points.begin(), p2 = points.begin() + 1; p2 != points.end();
       ++p1, ++p2) {
    line(img, *p1, *p2, line_colour);
  }
}
void draw_points(Mat img, const std::vector<Point> &points,
                 Scalar point_colour = {255, 0, 0}) {
  if (img.type() != CV_8UC3)
    return;
  for (const auto &p : points) {
    line(img, p, p, {255, 0, 0});
  }
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
  std::string r =
      fmt::format(std::locale("ru_RU.UTF-8"), "{};{:.2Lf};{:.2Lf};{:.2Lf}\n",
                  name, data.at("average_point"), data.at("average_deviation"),
                  data.at("max_deviation"));
  f << std::regex_replace(r, std::regex{"Â"}, "");
}

auto make_data(Mat img, const std::string &folder,
               const std::string &save_name) {
  auto line_points = find_line_points(img);
  write_data(folder, save_name, form_data(line_points));
  return line_points;
}

Mat crop_img(Mat img) {
  return img(Range(img.rows / 4, img.rows / 4 * 3),
             Range(img.cols / 4, img.cols / 4 * 3));
}

template <class F>
auto filter_img(Mat img, std::string folder, std::string save_name, F func) {
  Mat f_img = std::invoke(func, img);
  Mat e_f_img = detect_edges(f_img);
  Mat bw_f_img = cvt_to_bw(f_img);
  Mat e_bw_f_img = detect_edges(bw_f_img);

  save_image(f_img, folder, save_name + _STR(f_img));
  save_image(e_f_img, folder, save_name + _STR(e_f_img));
  save_image(bw_f_img, folder, save_name + _STR(bw_f_img));
  save_image(e_bw_f_img, folder, save_name + _STR(e_bw_f_img));

  auto line_points = make_data(bw_f_img, folder, save_name + _STR(bw_f_img));
  Mat img_with_line_and_points;
  cvtColor(img, img_with_line_and_points, COLOR_GRAY2BGR);
  draw_line(img_with_line_and_points, line_points);
  draw_points(img_with_line_and_points, line_points);

  save_image(img_with_line_and_points, folder, save_name + "_line");
  save_image(crop_img(img_with_line_and_points), folder,
             save_name + "_resized");

  return line_points;
}

void make_csv(const std::string &folder) {
  std::filesystem::path fullpath(folder);
  fullpath /= data_filename.data();
  std::ofstream f(fullpath);
  f << ";Average point;Average deviation;Max deviation\n";
}

void add_original_data(Mat img, const std::string &folder) {
  const std::string name = "original";
  Mat bw_img = cvt_to_bw(img);
  save_image(bw_img, folder, name + _STR(bw_img));
  const auto line_points = make_data(bw_img, folder, name);
  Mat img_with_line_and_points;
  cvtColor(img, img_with_line_and_points, COLOR_GRAY2BGR);
  draw_line(img_with_line_and_points, line_points);
  draw_points(img_with_line_and_points, line_points);
  save_image(img_with_line_and_points, folder, name + "_line");
  save_image(crop_img(img_with_line_and_points), folder, name + "_resized");
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    fmt::print("Not enough arguments. File path is needed\n");
    return EXIT_FAILURE;
  }

  Mat original = imread(argv[1]);
  Mat img;
  cvtColor(original, img, COLOR_RGB2GRAY);

  std::string folder = make_save_folder();
  make_csv(folder);
  add_original_data(img, folder);
  save_image(detect_edges(img), folder, "orignal_edges");

  Mat graph_morph_img = original.clone();
  draw_line(graph_morph_img, filter_img(img, folder, "dilate", &apply_dilate),
            {0, 128, 128});
  draw_line(graph_morph_img,
            filter_img(img, folder, "custom_closer", &apply_custom_closer),
            {255, 0, 0});
  draw_line(graph_morph_img, filter_img(img, folder, "closer", &apply_closer),
            {128, 0, 128});
  draw_line(graph_morph_img,
            filter_img(img, folder, "custom_opening", &apply_custom_opening),
            {0, 255, 0});
  draw_line(graph_morph_img, filter_img(img, folder, "erode", &apply_erode),
            {128, 128, 0});
  draw_line(graph_morph_img, filter_img(img, folder, "opening", &apply_opening),
            {0, 0, 255});
  save_image(crop_img(graph_morph_img), folder, STR(graph_morph_img));

  Mat graph_blur_img = original.clone();
  draw_line(
      graph_blur_img,
      filter_img(img, folder, "bilateral_filter", &apply_bilateral_filter),
      {0, 255, 0});
  draw_line(graph_blur_img,
            filter_img(img, folder, "median_blur", &apply_median_blur),
            {255, 0, 0});
  draw_line(graph_blur_img,
            filter_img(img, folder, "gaussian_blur", &apply_gaussian_blur),
            {0, 255, 255});
  draw_line(graph_blur_img, filter_img(img, folder, "blur", &apply_blur),
            {0, 0, 255});
  save_image(graph_blur_img, folder, STR(graph_blur_img));
}
