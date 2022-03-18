#include "morphological_processor.h"

sf::morphological_processor::morphological_processor() {
  parameter_values.emplace(parameter::kernel, Mat::ones(3, 3, CV_8UC1));
  parameter_values.emplace(parameter::anchor, Point{-1, -1});
  parameter_values.emplace(parameter::iteration_number, 3);
  parameter_values.emplace(parameter::custom_iteration_number, 1);
}
