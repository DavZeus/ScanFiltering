#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#define _STR(x) "_" #x
#define STR(x) #x
using namespace cv;

const std::string_view window_name = "1";
const std::string_view data_filename = "big_data.csv";

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

void save_image(Mat img, const std::string &path, const std::string &filename) {
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

Mat cvt_to_bw(Mat img, bool show = true) {
  Mat bw_img = img.clone();
  bw_img.forEach<uint8_t>([](uint8_t &p, const int *pos) {
    if (p != 255)
      p = 0;
  });
  if (show)
    show_image(bw_img);
  return bw_img;
}

std::vector<Point> find_line_points(Mat img) {
  std::vector<Point> line_points;
  line_points.reserve(img.cols);
  for (int i = 0; i < img.cols; ++i) {
    Mat_<uchar> col = img.col(i);
    auto first = std::find(col.begin(), col.end(), 255);
    auto last = std::find(col.rbegin(), col.rend(), 255);
    if (first != col.end() && last != col.rend()) {
      line_points.emplace_back(i, ((last.base().pos() + first.pos()) / 2).y);
    }
  }
  line_points.shrink_to_fit();
  return line_points;
}

void draw_line(Mat img, const std::string &folder, const std::string &save_name,
               std::vector<Point> points) {
  if (points.empty()) {
    return;
  }
  Mat img_with_line;
  cvtColor(img, img_with_line, COLOR_GRAY2BGR);
  for (auto p1 = points.begin(), p2 = points.begin() + 1; p2 != points.end();
       ++p1, ++p2) {
    line(img_with_line, *p1, *p2, {0, 0, 255});
  }
  // To draw only points
  // for (const auto &p : points) {
  //  line(img_with_line, p, p, {0, 0, 255});
  // }
  save_image(img_with_line, folder, save_name + "_line");
}

auto form_data(const std::vector<Point> &line_points) {
  const int average_point =
      std::reduce(line_points.begin(), line_points.end()).y /
      static_cast<int>(line_points.size());
  std::vector<int> deviation;
  deviation.reserve(line_points.size());
  for (auto p : line_points) {
    deviation.emplace_back(std::abs(p.y - average_point));
  }
  const int max_deviation =
      *std::max_element(deviation.begin(), deviation.end());
  const int average_deviation =
      std::reduce(deviation.begin(), deviation.end()) /
      static_cast<int>(deviation.size());
  std::map<std::string, int> data;
  data.emplace(STR(average_point), average_point);
  data.emplace(STR(max_deviation), max_deviation);
  data.emplace(STR(average_deviation), average_deviation);
  return data;
}

void write_data(const std::string &folder, std::map<std::string, int> data) {
  std::filesystem::path full_path(folder);
  full_path /= data_filename;
  std::ofstream f(full_path, std::ios::ate);
  if (!f) {
    fmt::print("Can not open data file\n");
    return;
  }
  f.imbue(std::locale("ru_RU.UTF-8"));
  for (const auto &[key, value] : data) {
    f << key << ": " << value << ';';
  }
  f << '\n';
}

// TODO: Rewrite to use map instead of vector
void make_data(Mat img, const std::string &folder,
               const std::string &save_name) {
  const std::vector<Point> line_points = find_line_points(img);
  if (line_points.empty()) {
    fmt::print("No points found in {}\n", save_name);
    return;
  }
  write_data(folder, save_name, form_data(line_points));
  draw_line(img, folder, save_name, line_points);
}

template <class F>
void filter_img(Mat img, std::string folder, std::string save_name, F func,
                bool show = true) {
  Mat bw_img = cvt_to_bw(img, show);
  Mat f_img = func(img, show);
  Mat e_f_img = detect_edges(f_img, show);
  Mat bw_f_img = cvt_to_bw(f_img, show);
  Mat e_bw_f_img = detect_edges(bw_f_img, show);
  // Mat otsu_f_img;
  // threshold(f_img, otsu_f_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
  // Mat e_otsu_f_img = detect_edges(bw_f_img, show);

  save_image(bw_img, folder, save_name + _STR(bw_img));
  save_image(f_img, folder, save_name + _STR(f_img));
  save_image(e_f_img, folder, save_name + _STR(e_f_img));
  save_image(bw_f_img, folder, save_name + _STR(bw_f_img));
  save_image(e_bw_f_img, folder, save_name + _STR(e_bw_f_img));
  // save_image(otsu_f_img, folder, save_name + _STR(otsu_f_img));
  // save_image(e_otsu_f_img, folder, save_name + _STR(e_otsu_f_img));

  make_data(bw_img, folder, save_name + _STR(bw_img));
  make_data(bw_f_img, folder, save_name + _STR(bw_f_img));
  // avg(otsu_f_img, folder, save_name + _STR(otsu_f_img));
}

void make_csv(const std::string &folder) {
  std::filesystem::path fullpath(folder);
  fullpath /= data_filename.data();
  std::ofstream f(fullpath);
  f << ";Average point;Max deviation;Average deviation\n";
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
  make_csv(folder);
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
