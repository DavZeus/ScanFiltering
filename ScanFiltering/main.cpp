#include <opencv2/opencv.hpp>
#include <numbers>
#include <fmt/core.h>
#include <array>

using namespace cv;

const std::string_view window_name = "1";

void show_image(Mat img) {
	imshow(window_name.data(), img);
	waitKey(0);
}

template<class F, class ...Args>
Mat apply_filter(Mat img, bool show, F func, Args... args) {
	Mat filtered_img;
	func(img, filtered_img, std::forward<Args>(args)...);
	if (show) show_image(filtered_img);
	return filtered_img;
}

Mat detect_edges(Mat img, bool show = true) {
	Mat edges;
	const double low_threshold = 240;
	const double high_threshhold = 250;
	Canny(img, edges, low_threshold, high_threshhold, 3);
	if (show) show_image(edges);
	return edges;
}

Mat cvt_non_white_to_black(Mat img, bool show = true) {
	Mat bw_img = img.clone();
	bw_img.forEach<uint8_t>([](uint8_t& p, const int* pos) {
		if (p != 255) p = 0;
	});
	if (show) show_image(bw_img);
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

	std::array filtered_images = {
		//apply_filter(gray_img, true, &filter2D, gray_img.type(), Mat::ones(3, 3, CV_32F) / static_cast<float>(9), Point(-1, -1), 0, BORDER_DEFAULT),
		//apply_filter(gray_img, true, &blur, Size(3, 3), Point(-1, -1), BORDER_DEFAULT),
		//apply_filter(gray_img, true, &boxFilter, gray_img.type(), Size(3, 3), Point(-1, -1), true, BORDER_DEFAULT),
		//apply_filter(gray_img, true, &bilateralFilter, 9, 50, 50, BORDER_DEFAULT),
		//apply_filter(gray_img, true, &GaussianBlur, cv::Size(3, 3), 0, 0, BORDER_DEFAULT),
		//apply_filter(gray_img, true, &medianBlur, 3),
		apply_filter(gray_img, 
								 true,
								 &dilate,
								 Mat::ones(3, 3, gray_img.type()),
								 Point(-1, -1),
								 3,
								 BORDER_CONSTANT,
								 morphologyDefaultBorderValue()
		),
		apply_filter(gray_img, 
								 true,
								 &erode,
								 Mat::ones(3, 3, gray_img.type()),
								 Point(-1, -1),
								 3,
								 BORDER_CONSTANT,
								 morphologyDefaultBorderValue()
		),
		apply_filter(gray_img,
								 true,
								 &morphologyEx,
								 MORPH_CLOSE,
								 Mat::ones(3, 3, gray_img.type()),
								 Point(-1, -1),
							   1,
								 BORDER_CONSTANT,
							   morphologyDefaultBorderValue()
		),
		apply_filter(gray_img,
								 true,
								 &morphologyEx,
								 MORPH_OPEN,
								 Mat::ones(3, 3, gray_img.type()),
								 Point(-1, -1),
							   2,
								 BORDER_CONSTANT,
							   morphologyDefaultBorderValue()
		),
	};

	for (const auto img : filtered_images) {
		detect_edges(img);
		//const auto bw_img = cvt_non_white_to_black(img);
		//detect_edges(bw_img);
	}
}