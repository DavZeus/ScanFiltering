#include "drawer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace sf {
Scalar drawer::get_color(size_t i) {
  return color_priority.at(i > color_priority.size() ? color_priority.size() - 1
                                                     : i);
}
Scalar drawer::get_best_color() { return *color_priority.rbegin(); }
void drawer::save_image(Mat img, const std::string &filename) {
  if (img.empty())
    return;
  auto file_path = folder / (common_name + '_' + filename + ".png");
  imwrite(file_path.string(), img);
}
void drawer::draw_line(Mat img, const line &points, Scalar line_color) {
  // TODO: Remove
  if (img.type() != CV_8UC3 || points.empty())
    return;
  for (auto p1 = points.begin(), p2 = points.begin() + 1; p2 != points.end();
       ++p1, ++p2) {
    cv::line(img, *p1, *p2, line_color);
  }
}
Mat drawer::crop_img(Mat img) {
  const float first_x = center_x - shift_x;
  const float second_x = center_x + shift_x;
  const float first_y = center_y - shift_y;
  const float second_y = center_y + shift_y;
  const int x0 = static_cast<int>(img.rows * first_y);
  const int x1 = static_cast<int>(img.rows * second_y);
  const int y0 = static_cast<int>(img.cols * first_x);
  const int y1 = static_cast<int>(img.cols * second_x);
  return img(Range(x0, x1), Range(y0, y1));
}
Mat drawer::detect_edges(Mat img) {
  Mat edges;
  const double low_threshold = 254;
  const double high_threshhold = 255;
  Canny(img, edges, low_threshold, high_threshhold, 3);
  return edges;
}
void drawer::make_edge_imgs(Mat original, const map_of_images &imgs) {
  save_image(detect_edges(original), "original_edge");
  for (const auto &[name, img] : imgs) {
    Mat edge_img = detect_edges(img);
    save_image(edge_img, std::string(name) + "_edge");
  }
}
Mat drawer::form_single_line_image(Mat img, const line &points) {
  Mat img_with_line = img.clone();
  draw_line(img_with_line, points, get_best_color());
  return img_with_line;
}
void drawer::make_line_imgs(const map_of_images &imgs,
                            const map_of_lines &lines) {
  for (const auto &[filter_name, img] : imgs) {
    Mat img_with_line = form_single_line_image(img, lines.at(filter_name));
    save_image(img_with_line, std::string(filter_name) + "_line");
    save_image(crop_img(img_with_line),
               std::string(filter_name) + "_line_crop");
  }
}
void drawer::make_graph_img(Mat original, const map_of_lines &lines) {
  Mat graph_img = original.clone();
  size_t i = 0;
  for (const auto &[filter_name, points] : lines) {
    draw_line(graph_img, points, get_color(i));
  }
  save_image(graph_img, "graph");
}
void drawer::set_folder(std::string folder) {
  this->folder = std::move(folder);
}
void drawer::set_common_name(std::string name) {
  common_name = std::move(name);
}
void drawer::set_crop(float center_x, float center_y, float shift_x,
                      float shift_y) {
  this->center_x = center_x;
  this->center_y = center_y;
  this->shift_x = shift_x;
  this->shift_y = shift_y;
}
void drawer::make_info_images(Mat original, const map_of_images &imgs,
                              const map_of_lines &lines) {
  make_edge_imgs(original, imgs);
  make_line_imgs(imgs, lines);
  make_graph_img(original, lines);
}
} // namespace sf
