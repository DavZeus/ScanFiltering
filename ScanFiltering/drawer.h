#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <opencv2/core/mat.hpp>
#include <string>

namespace sf {
using namespace cv;
class drawer {
  using line = std::vector<Point>;
  using map_of_lines = std::map<std::string_view, line>;
  using map_of_images = std::map<std::string_view, Mat>;

  // Priority increases with color index
  inline static const std::array<Scalar, 6> color_priority{
      Scalar{0, 128, 255}, {0, 234, 255}, {0, 128, 0},
      {255, 0, 0},         {179, 0, 179}, {0, 0, 255}};

  Scalar get_color(size_t i);
  Scalar get_best_color();

  float center_x = 0.535f;
  float center_y = 0.5f;
  float shift_x = 0.05f;
  float shift_y = 0.03f;

  std::string common_name;
  std::filesystem::path folder;
  void save_image(Mat img, const std::string &name);

  void draw_line(Mat img, const line &points, Scalar line_color = {0, 0, 255});

  struct coordinates {
    int x0, x1, y0, y1;
  };
  coordinates calculate_crop_coords(Mat img);
  Mat crop_img(Mat img);
  void make_crop_rectangle(Mat img);

  Mat detect_edges(Mat img);
  void make_edge_imgs(Mat original, const map_of_images &imgs);

  Mat form_single_line_image(Mat img, const line &points);
  void make_line_imgs(const map_of_images &imgs, const map_of_lines &lines);

  void make_graph_img(Mat original, const map_of_lines &lines);

public:
  void set_folder(std::filesystem::path folder);
  void set_common_name(std::string name);
  void set_crop(float center_x, float center_y, float shift_x, float shift_y);
  void make_info_images(Mat original, const map_of_images &imgs,
                        const map_of_lines &lines);
};
} // namespace sf
