#include <opencv2/opencv.hpp>
#include <iostream>
#include "detect.hpp"

using namespace std;

float LINE_EQUAL_DEGREE;         // 同じ線分とみなす線分間の最大角度
float LINE_CROSS_DEGREE;         // 直交とみなす最小角度
float LINE_EQUAL_DISTANCE;      // 同じ線分とみなす中点同士の最大垂直距離
float POINT_CONNECT_DISTANCE;  // 同じ線分とみなす端点の最大距離
float CENTER_WIDTH;               // 画像中心部の幅
float CENTER_HEIGHT;              // 画像中心部の高さ
float INTERSECT_DIST_RATIO;     // 線分から交点までの距離の最大値(割合)

int POINT_IN_SECTION;             // 画像の4箇所それぞれが含む候補点の数

const float EPSILON = 0.0;  // 1e-7
const float F_PI = 3.1415926f;
const float SQRT2 = 1.4142135f;
const float INF = 1e7;

cv::Size img_size;
int img_avglen;

void drawLines(cv::Mat& img, vector<cv::Vec4f>& lines) {
	for (int i = 0; i < lines.size(); i++) {
		cv::line(img, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0));
		cv::putText(img, to_string(i),
				cv::Point((lines[i][0]+lines[i][2])*0.5, (lines[i][1]+lines[i][3])*0.5),
				cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(126));		
	}
}

int main(int argc, char const *argv[])
{
	cv::Mat img;
	img.create(800, 600, CV_8UC1);
	img = cv::Scalar(255);

	init(img);

	vector<cv::Vec4f> lines;
	lines.push_back(cv::Vec4f(81.945374, 755.711670, 427.026917, 762.663513));
	lines.push_back(cv::Vec4f(280.000000, 761.000000, 236.000000, 761.000000));

	drawLines(img, lines);

	// bool ans = line_equal(line1, line2);
	// cout << "line equal ? " << ans << endl;
	for (int i = 0; i < lines.size(); i++) {
		printf("17, %d: %d\n", i, line_equal(lines[17], lines[i]));
	}

	std::vector<int> labels;
  int labels_num = cv::partition(lines, labels, line_equal);

  for (int i = 0; i < lines.size(); i++) {
  	cout << i << " " << labels[i] << " " << lines[i] << endl;
  }

	cv::imshow("image", img);
	cv::waitKey();

	return 0;
}