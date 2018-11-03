#include <opencv2/opencv.hpp>
#include <iostream>
#include "detect.hpp"

using namespace std;

float LINE_EQUAL_DEGREE;         // 同じ線分とみなす線分間の最大角度
float LINE_EQUAL_DISTANCE;      // 同じ線分とみなす中点同士の最大垂直距離
float POINT_EQUAL_DISTANCE;     // 別の線分の端点を同じ点とみなす最大距離
float LINE_INCLUDE_DISTANCE;    // 線分に点が含まれるとみなす最大距離
float LINE_CROSS_DEGREE;         // 直交とみなす最小角度
float CENTER_WIDTH;               // 画像中心部の幅
float CENTER_HEIGHT;              // 画像中心部の高さ

int POINT_IN_SECTION;             // 画像の4箇所それぞれが含む候補点の数

const float EPSILON = 0.0;  // 1e-7
const float F_PI = 3.1415926f;
const float SQRT2 = 1.4142135f;
const float INF = 1e7;

cv::Size img_size;

void drawLine(cv::Mat& img, cv::Vec4f& line) {
	cv::line(img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0));
}

int main(int argc, char const *argv[])
{
	cv::Mat img;
	img.create(600, 450, CV_8UC1);
	img = cv::Scalar(255);

	init(img);

	vector<cv::Vec4f> lines;
	lines.push_back(cv::Vec4f(24.9915, 578.901, 398.896, 546.788));
	lines.push_back(cv::Vec4f(378.987, 551.781, 352.021, 553.362));
	lines.push_back(cv::Vec4f(264.837, 559.509, 32.1048, 584.959));
	lines.push_back(cv::Vec4f(297.018, 575.333, 319.008, 574.144));
	lines.push_back(cv::Vec4f(346.182, 599, 443.072, 587.586));
	lines.push_back(cv::Vec4f(351.071, 593.742, 301.054, 598.56));

	for (int i = 0; i < lines.size(); i++)
			drawLine(img, lines[i]);

	// bool ans = line_equal(line1, line2);
	// cout << "line equal ? " << ans << endl;

	std::vector<int> labels;
  int labels_num = cv::partition(lines, labels, line_equal);

  for (int i = 0; i < lines.size(); i++) {
  	cout << labels[i] << " " << lines[i] << endl;
  }

	cv::imshow("image", img);
	cv::waitKey();

	return 0;
}