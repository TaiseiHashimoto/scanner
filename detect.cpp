#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <cmath>
#include <cassert>
#include "detect.hpp"

#define DISPLAY_SIZE 600  // 表示する際の高さ

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
  cv::namedWindow("original image", cv::WINDOW_NORMAL);
  cv::imshow("original image", image);

  cv::Mat processed;

  // Create FLD detector
  int length_threshold = (img_size.width + img_size.height) / 2 * 0.04; //20;
  float distance_threshold = (img_size.width + img_size.height) / 2 * 0.003;// 1.41421356f;
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
  cout << "num of lines = " << lines.size() << " => ";
  remove_central_lines(lines);
  cout << lines.size() << endl;

  std::vector<int> labels;
  int labels_num = cv::partition(lines, labels, line_equal);

  // Show found lines
  draw_labeled_lines(image, processed, lines, labels, labels_num);
  cv::namedWindow("line detected", cv::WINDOW_NORMAL);
  cv::imshow("line detected", processed);

  vector<Segment> segments;
  get_segments(lines, segments, labels, labels_num);
  cout << "num of segments: " << segments.size() << " => ";
  remove_central_segments(segments);
  cout << segments.size() << endl;

  draw_lines(image, processed, segments);
  cv::namedWindow("line refined", cv::WINDOW_NORMAL);
  cv::imshow("line refined", processed);

  vector<Intersection> intersections;
  Intersection::get_intersections(segments, intersections);

  cv::cvtColor(image, processed, cv::COLOR_GRAY2BGR);
  sort(intersections.begin(), intersections.end(), 
        [](const Intersection& left, const Intersection& right) {
          return left.get_score() > right.get_score();
        });

  for (int i = 0; i < intersections.size() && i < 40; i++) {
    Intersection& inter = intersections[i];
    cv::Point2f cross_point = inter.get_cross_point();
    cv::putText(processed, to_string(inter.m_id), cross_point, cv::FONT_HERSHEY_PLAIN,
                   1.0, cv::Scalar(255, 0, 0));
    cv::circle(processed, cross_point, 2, cv::Scalar(255, 0, 0), -1);
    // printf("%s", inter.m_description);
  }
  cout << "num of intersections " << intersections.size() << endl;
  cv::imshow("intersections detected", processed);

  vector<Intersection> inters_lt, inters_rt, inters_lb, inters_rb;
  for (int i = 0; i < intersections.size(); i++) {
    Intersection& inter = intersections[i];
    if (inter.is_top()) {
      if (inter.is_left() && inters_lt.size() < POINT_IN_SECTION) {
        inters_lt.push_back(inter);
      } else if (inter.is_right() && inters_rt.size() < POINT_IN_SECTION) {
        inters_rt.push_back(inter);
      }
    } else {
      if (inter.is_left() && inters_lb.size() < POINT_IN_SECTION) {
        inters_lb.push_back(inter);
      } else if (inter.is_right() && inters_rb.size() < POINT_IN_SECTION) {
        inters_rb.push_back(inter);
      }
    }
  }

  vector<vector<int> > indice;
  get_combi_indice(inters_lt.size(), inters_rt.size(),
                        inters_lb.size(), inters_rb.size(), indice);

  // どれか1つでも空の場合はエラー
  assert(!inters_lt.empty());
  assert(!inters_lt.empty());
  assert(!inters_lt.empty());
  assert(!inters_lt.empty());
  float best_score = -INF;
  int idx_lt, idx_rt, idx_lb, idx_rb;
  for (int i = 0; i < indice.size(); i++) {
    Intersection& inter_lt = inters_lt[indice[i][0]];
    Intersection& inter_rt = inters_rt[indice[i][1]];
    Intersection& inter_lb = inters_lb[indice[i][2]];
    Intersection& inter_rb = inters_rb[indice[i][3]];

    float score = (inter_lt.get_score() + inter_rt.get_score()
                    + inter_lb.get_score() + inter_rb.get_score()) / 4;

    cv::Point2f v1 = inter_rt.get_cross_point() - inter_lt.get_cross_point();
    cv::Point2f v2 = inter_rb.get_cross_point() - inter_rt.get_cross_point();
    cv::Point2f v3 = inter_lb.get_cross_point() - inter_rb.get_cross_point();
    cv::Point2f v4 = inter_lt.get_cross_point() - inter_lb.get_cross_point();
    float deg1 = acosf(v4.dot(v1) / cv::norm(v4) / cv::norm(v1));
    float deg2 = acosf(v1.dot(v2) / cv::norm(v1) / cv::norm(v2));
    float deg3 = acosf(v2.dot(v3) / cv::norm(v2) / cv::norm(v3));
    float deg4 = acosf(v3.dot(v4) / cv::norm(v3) / cv::norm(v4));

    float weight = 500;
    score -= (powf(angle_sub(deg1, F_PI/2), 2) + powf(angle_sub(deg2, F_PI/2), 2)
              + powf(angle_sub(deg1, F_PI/2), 2) + powf(angle_sub(deg2, F_PI/2), 2)) * weight;

    if (best_score < score) {
      best_score = score;
      idx_lt = indice[i][0];
      idx_rt = indice[i][1];
      idx_lb = indice[i][2];
      idx_rb = indice[i][3];
    }
    // printf("(%d, %d, %d, %d) : %f\n", inter_lt.m_id, inter_rt.m_id, inter_lb.m_id, inter_rb.m_id, score);
  }

  cv::cvtColor(image, processed, cv::COLOR_GRAY2BGR);
  cv::circle(processed, inters_lt[idx_lt].get_cross_point(), 3, cv::Scalar(0, 0, 255), -1);
  cv::circle(processed, inters_rt[idx_rt].get_cross_point(), 3, cv::Scalar(0, 0, 255), -1);
  cv::circle(processed, inters_lb[idx_lb].get_cross_point(), 3, cv::Scalar(0, 0, 255), -1);
  cv::circle(processed, inters_rb[idx_rb].get_cross_point(), 3, cv::Scalar(0, 0, 255), -1);
  cv::line(processed, inters_lt[idx_lt].get_cross_point(), inters_rt[idx_rt].get_cross_point(), cv::Scalar(0, 0, 255));
  cv::line(processed, inters_rt[idx_rt].get_cross_point(), inters_rb[idx_rb].get_cross_point(), cv::Scalar(0, 0, 255));
  cv::line(processed, inters_rb[idx_rb].get_cross_point(), inters_lb[idx_lb].get_cross_point(), cv::Scalar(0, 0, 255));
  cv::line(processed, inters_lb[idx_lb].get_cross_point(), inters_lt[idx_lt].get_cross_point(), cv::Scalar(0, 0, 255));

  printf("best score: %f  (%d, %d, %d, %d)\n", best_score,
      inters_lt[idx_lt].m_id,  inters_rt[idx_rt].m_id, inters_lb[idx_lb].m_id, inters_rb[idx_rb].m_id);
  cv::namedWindow("rectangle detected", cv::WINDOW_NORMAL);
  cv::imshow("rectangle detected", processed);

  vector<cv::Point2f> src_points = {inters_lt[idx_lt].get_cross_point(),
                                          inters_rt[idx_rt].get_cross_point(),
                                          inters_lb[idx_lb].get_cross_point(),
                                          inters_rb[idx_rb].get_cross_point()};
  vector<cv::Point2f> dst_points = {cv::Point2f(0, 0),
                                          cv::Point2f(img_size.width, 0),
                                          cv::Point2f(0, img_size.height),
                                          cv::Point2f(img_size.width, img_size.height)};
  cv::Mat M = findHomography(src_points, dst_points, cv::RANSAC, 3);
  cv::warpPerspective(image, processed, M, img_size);
  cv::namedWindow("homography result", cv::WINDOW_NORMAL);
  cv::imshow("homography result", processed);

  cv::imwrite("detect_out.jpeg", processed);

  double duration_ms = (double(cv::getTickCount()) - start) * 1000 / cv::getTickFrequency();
  cout << "It took " << duration_ms << " ms." << endl;

  cv::waitKey();

  return 0;
}
