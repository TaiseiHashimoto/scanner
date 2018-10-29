#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <cmath>
#include <cassert>
#include "detect.hpp"

using namespace std;

float LINE_EQUAL_DEGREE;         // 同じ線分とみなす線分間の最大角度
float LINE_EQUAL_DISTANCE;      // 同じ線分とみなす中点同士の最大垂直距離
float POINT_EQUAL_DISTANCE;     // 別の線分の端点を同じ点とみなす最大距離
float LINE_INCLUDE_DISTANCE;    // 線分に点が含まれるとみなす最大距離
float LINE_CROSS_DEGREE;         // 直交とみなす最小角度
float CENTER_WIDTH;               // 画像中心部の幅
float CENTER_HEIGHT;              // 画像中心部の高さ

const float EPSILON = 0.0;  // 1e-7
const float F_PI = 3.1415926f;
const float SQRT2 = 1.4142135f;

cv::Size img_size;

int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, "{@input|../opencv-3.2.0/samples/data/building.jpg|input image}{help h||show help message}");
  if (parser.has("help")) {
      parser.printMessage();
      return 0;
  }
  string in = parser.get<string>("@input");

  cv::Mat image = cv::imread(in, cv::IMREAD_GRAYSCALE);

  if (image.empty()) { 
    return -1;
  }

  double start = double(cv::getTickCount());

  init(image);
  // exit(0);

  // Create FLD detector
  int length_threshold = 20;
  float distance_threshold = 1.41421356f;
  double canny_th1 = 5.0;
  double canny_th2 = 50.0;
  int canny_aperture_size = 3;
  bool do_merge = false;
  cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(
    length_threshold,
    distance_threshold, canny_th1, canny_th2, canny_aperture_size,
    do_merge
  );

  vector<cv::Vec4f> lines;
  fld->detect(image, lines);

  cout << "num of lines = " << lines.size() << endl;

  std::vector<int> labels;
  int labels_num = cv::partition(lines, labels, line_equal);
  printf("labels_num = %d\n", labels_num);

  // Show found lines
  cv::Mat drawnLines;
  // fld->drawSegments(drawnLines, lines);
  draw_labeled_lines(image, drawnLines, lines, labels, labels_num);
  cv::imshow("line detected", drawnLines);

  vector<Segment> segments;
  get_segments(lines, segments, labels, labels_num);
  cout << "num of segments = " << segments.size() << endl;
  // cv::Mat mergedLines;
  // draw_lines(image, mergedLines, segments);
  // cv::imshow("line merged", mergedLines);
  
  remove_central_segments(segments);
  cout << "num of segments = " << segments.size() << endl;
  cv::Mat refinedLines;
  draw_lines(image, refinedLines, segments);
  cv::imshow("line refined", refinedLines);

  vector<cv::Point2f> inter_points;
  for (int i = 0; i < segments.size(); i++) {
    for (int j = i + 1; j < segments.size(); j++) {
      cv::Point2f inter_point;
      int type = segments[i].intersection(segments[j], inter_point);
      if (type < 0) continue;
      inter_points.push_back(inter_point);
    }
  }
  cv::Mat interPoints;
  cv::cvtColor(image, interPoints, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < inter_points.size(); i++) {
    cv::circle(interPoints, inter_points[i], 3, cv::Scalar(255, 0, 0), -1);
  }
  cout << "num of intersections " << inter_points.size() << endl;
  cv::imshow("intersections detected", interPoints);

  double duration_ms = (double(cv::getTickCount()) - start) * 1000 / cv::getTickFrequency();
  cout << "It took " << duration_ms << " ms." << endl;

  cv::waitKey();


  return 0;
}
