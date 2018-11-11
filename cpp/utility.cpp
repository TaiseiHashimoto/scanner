#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include "detect.hpp"

using namespace std;

void init(cv::Mat& image) {
  // 各種値を初期化
  img_size = image.size();
  img_avglen = (img_size.width + img_size.height) / 2;
  cout << img_size << endl;
  LINE_EQUAL_DEGREE = 2;
  LINE_CROSS_DEGREE = 60;
  LINE_EQUAL_DISTANCE = img_avglen * 0.005;    // 500 =>  2.5
  POINT_CONNECT_DISTANCE = img_avglen * 0.1;   // 500 => 50
  CENTER_WIDTH = img_size.width * 0.5;
  CENTER_HEIGHT = img_size.height * 0.5;
  INTERSECT_DIST_RATIO = 0.5;

  POINT_IN_SECTION = 5;

  cout << "LINE_EQUAL_DEGREE " << LINE_EQUAL_DEGREE << endl;
  cout << "LINE_CROSS_DEGREE " << LINE_CROSS_DEGREE << endl;
  cout << "LINE_EQUAL_DISTANCE " << LINE_EQUAL_DISTANCE << endl;
  cout << "POINT_CONNECT_DISTANCE " << POINT_CONNECT_DISTANCE << endl;
  cout << "CENTER_WIDTH " << CENTER_WIDTH << endl;
  cout << "CENTER_HEIGHT " << CENTER_HEIGHT << endl;
  cout << "INTERSECT_DIST_RATIO " << INTERSECT_DIST_RATIO << endl;
}

float angle_sub(float theta1, float theta2) {
  if (theta1 - theta2 < -F_PI/2) {
    return theta1 - theta2 + F_PI;
  } else if (theta1 - theta2 > F_PI/2) {
    return theta1 - theta2 - F_PI;
  }
  return theta1 - theta2;
}

void remove_central_lines(vector<cv::Vec4f>& lines) {
  vector<cv::Vec4f> lines_refined;
  for (int i = 0; i < lines.size(); i++) {
    cv::Vec4f& line = lines[i];
    float mx = (line[0] + line[2]) / 2;
    float my = (line[1] + line[3]) / 2;
    if (fabs(mx - img_size.width/2) > CENTER_WIDTH/2 ||
        fabs(my - img_size.height/2) > CENTER_HEIGHT/2) {
      lines_refined.push_back(line);
    }
  }
  lines = lines_refined;
}

void remove_central_segments(vector<Segment>& segments) {
  vector<Segment> segments_refined;
  for (int i = 0; i < segments.size(); i++) {
    Segment& seg = segments[i];
    cv::Point2f mu = (seg.m_pe1 + seg.m_pe2) / 2.0;
    if (fabs(mu.x - img_size.width/2) > CENTER_WIDTH/2 || 
        fabs(mu.y - img_size.height/2) > CENTER_HEIGHT/2) {
      segments_refined.push_back(seg);
    }
  }
  segments = segments_refined;
}

void draw_lines(cv::Mat& src, cv::Mat& dst, vector<Segment>& segments) {
  cv::RNG rng(12345);
  cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
  vector<cv::Scalar> colors;
  for (int i = 0; i < segments.size(); i++) {
    Segment& seg = segments[i];
    // if (seg.m_id > 100) continue;
    cv::Scalar color(rng.uniform(0, 127), rng.uniform(0, 127), rng.uniform(0, 127));
    colors.push_back(color);
    cv::line(dst, seg.m_p1, seg.m_p2, color*2, 2);
    cv::putText(dst, to_string(seg.m_id), seg.m_pm + cv::Point2f(2, -3), cv::FONT_HERSHEY_PLAIN, 1.0, colors[i]*2);
  }
}

void draw_labeled_lines(cv::Mat& src, cv::Mat& dst, vector<cv::Vec4f>& lines, vector<int> labels, int labels_num) {
  cv::RNG rng(123);
  vector<cv::Scalar> colors;
  for (int i = 0; i < labels_num; i++) {
    colors.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
  }

  cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);

  for (int i = 0; i < lines.size(); i++) {
    cv::Vec4f line = lines[i];
    cv::Scalar color = colors[labels[i]];
    cv::line(dst, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), color, 3);
  }
}

bool line_equal(const cv::Vec4f& l1, const cv::Vec4f& l2) {
  float theta1, theta2, theta_delta, theta_avg;
  theta1 = atanf((l1[1] - l1[3]) / (l1[0] - l1[2] + EPSILON));
  theta2 = atanf((l2[1] - l2[3]) / (l2[0] - l2[2] + EPSILON));
  if (theta1 < theta2) swap(theta1, theta2);
  if (theta1 - theta2 < F_PI / 2) {
    theta_delta = theta1 - theta2;
    theta_avg = (theta1 + theta2) / 2;
  } else {
    theta_delta = theta2 - theta1 + F_PI;
    theta_avg = (theta1 + theta2 + F_PI) / 2;
    if (theta_avg > F_PI / 2) theta_avg -= F_PI;
  }

  // 向きが異なる2線分はマージしない
  if (theta_delta * 180 > LINE_EQUAL_DEGREE * F_PI) {
    return false;
  }

  float dist_point = min({powf(l1[0] - l2[0], 2) + powf(l1[1] - l2[1], 2),
                              powf(l1[0] - l2[2], 2) + powf(l1[1] - l2[3], 2),
                              powf(l1[2] - l2[0], 2) + powf(l1[3] - l2[1], 2),
                              powf(l1[2] - l2[2], 2) + powf(l1[3] - l2[3], 2)});
  dist_point = sqrtf(dist_point);
  if (dist_point > POINT_CONNECT_DISTANCE) {
    return false;
  }

  float mx1 = (l1[0] + l1[2]) / 2;
  float my1 = (l1[1] + l1[3]) / 2;
  float mx2 = (l2[0] + l2[2]) / 2;
  float my2 = (l2[1] + l2[3]) / 2;
  float dist_ver = fabs(sinf(theta_avg) * (mx1 - mx2) - cosf(theta_avg) * (my1 - my2));
  if (dist_ver > LINE_EQUAL_DISTANCE) {
    return false;
  }

  return true;
}

void get_combi_indice(int am, int bm, int cm, int dm, vector<vector<int> >& indice) {
  for (int a = 0; a < am; a++) {
    for (int b = 0; b < bm; b++) {
      for (int c = 0; c < cm; c++) {
        for (int d = 0; d < dm; d++) {
          indice.push_back(vector<int>{a, b, c, d});
        }
      }
    }
  }
}

void imshow_resized(string window_name, cv::Mat& img, cv::Size& size) {
  cv::Mat resized;
  cv::resize(img, resized, size);
  cv::imshow(window_name, resized);
}