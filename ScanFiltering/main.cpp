#include <opencv2/opencv.hpp>
#include <numbers>
#include <fmt/core.h>
#include <array>

const std::string_view window_name = "1";

void show_image(cv::Mat img) {
	cv::imshow(window_name.data(), img);
	cv::waitKey(0);
}

template<class F, class ...Args>
cv::Mat apply_filter(cv::Mat img, bool show, F func, Args... args) {
	cv::Mat filtered_img;
	func(img, filtered_img, std::forward<Args>(args)...);
	if (show) show_image(filtered_img);
	return filtered_img;
}


//cv::Mat apply_blur(cv::Mat img, bool show = true) {
//	cv::Mat filtered_img;
//	cv::blur(img, filtered_img, );
//	if (show) show_image(filtered_img);
//	return filtered_img;
//}
//
//cv::Mat apply_box_filter(cv::Mat img, bool show = true) {
//	cv::Mat filtered_img;
//	cv::boxFilter(img, filtered_img, );
//	if (show) show_image(blur_img);
//	show_image(filtered_img);
//	return filtered_img;
//}
//
//cv::Mat apply_median_blur(cv::Mat img, bool show = true) {
//	cv::Mat filtered_img;
//	cv::medianBlur(img, filtered_img, );
//	if (show) show_image(blur_img);
//	show_image(filtered_img);
//	return filtered_img;
//}
//
//cv::Mat apply_gaussian_blur(cv::Mat img, bool show = true) {
//	cv::Mat filtered_img;
//	const int kernel = 9;
//	cv::GaussianBlur(img, filtered_img, { kernel, kernel }, 0);
//	if (show) show_image(blur_img);
//	show_image(filtered_img);
//	return filtered_img;
//}
//
//cv::Mat apply_bilateral_blur(cv::Mat img, bool show = true) {
//	cv::Mat bilateral_img;
//	int d = 9;
//	double sigma = 50;
//	cv::bilateralFilter(img, bilateral_img, d, sigma, sigma);
//	if (show) show_image(blur_img);
//	show_image(bilateral_img);
//	return bilateral_img;
//}
//
//cv::Mat apply_erode(cv::Mat img, bool show = true) {
//	cv::Mat erode_img;
//	cv::erode(img, erode_img, );
//	if (show) show_image(blur_img);
//	show_image(erode_img);
//	return erode_img;
//}
//
//cv::Mat apply_dilate(cv::Mat img, bool show = true) {
//	cv::Mat dilate_img;
//	cv::dilate(img, dilate_img, );
//	if (show) show_image(blur_img);
//	return dilate_img;
//}
//
cv::Mat detect_edges(cv::Mat img, bool show = true) {
	cv::Mat edges;
	const double low_threshold = 240;
	const double high_threshhold = 250;
	cv::Canny(img, edges, low_threshold, high_threshhold, 3);
	if (show) show_image(edges);
	return edges;
}

cv::Mat cvt_non_white_to_black(cv::Mat img, bool show = true) {
	cv::Mat bw_img = img.clone();
	bw_img.forEach<uint8_t>([](uint8_t& p, const int* pos) {
		if (p != 255) p = 0;
		});
	if (show) show_image(bw_img);
	return bw_img;
}

int main() {
	cv::namedWindow(window_name.data(), cv::WindowFlags::WINDOW_KEEPRATIO);

	cv::Mat img = cv::imread("20k_edge.bmp");
	show_image(img);

	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);

	std::array filtered_images = {
		apply_filter(img, true, &cv::blur, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_DEFAULT),
		apply_filter(img, true, &cv::bilateralFilter, 9, 50, 50, cv::BORDER_DEFAULT),
	};

	detect_edges(gray_img);

	/*const double rho = 1;
	constexpr double theta = std::numbers::pi / 180;
	const int threshold = 15;
	const double min_line_length = 5;
	const double max_line_gap = std::min(img.size().height, img.size().width) / 100.;
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(edges, lines, rho, theta, threshold, min_line_length, max_line_gap);
	for (const auto& line : lines) {
		const auto [x0, y0, x1, y1] = line.val;
		cv::line(img, { x0, y0 }, { x1, y1 }, { 255, 0, 0 }, 1);
	}
	cv::imshow(window_name.data(), img);
	cv::waitKey(0);*/
}