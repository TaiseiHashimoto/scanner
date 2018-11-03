#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include "detect.hpp"

using namespace std;

void init(cv::Mat& image) {
  // 各種値を初期化
  img_size = image.size();
  cout << img_size << endl;
  int len_avg = (img_size.width + img_size.height) / 2;
  LINE_EQUAL_DEGREE = 3;
  LINE_EQUAL_DISTANCE = len_avg * 0.005;    // 500 =>  2.5
  POINT_EQUAL_DISTANCE = len_avg * 0.006;   // 500 => 3.0
  LINE_INCLUDE_DISTANCE = len_avg * 0.03;  // 500 => 15.0
  LINE_CROSS_DEGREE = 60;
  CENTER_WIDTH = img_size.width * 0.6;
  CENTER_HEIGHT = img_size.height * 0.6;

  POINT_IN_SECTION = 5;

  cout << "LINE_EQUAL_DEGREE " << LINE_EQUAL_DEGREE << endl;
  cout << "LINE_EQUAL_DISTANCE " << LINE_EQUAL_DISTANCE << endl;
  cout << "POINT_EQUAL_DISTANCE " << POINT_EQUAL_DISTANCE << endl;
  cout << "LINE_INCLUDE_DISTANCE " << LINE_INCLUDE_DISTANCE << endl;
  cout << "LINE_CROSS_DEGREE " << LINE_CROSS_DEGREE << endl;
  cout << "CENTER_WIDTH " << CENTER_WIDTH << endl;
  cout << "CENTER_HEIGHT " << CENTER_HEIGHT << endl;
}

float angle_sub(float theta1, float theta2) {
  if (theta1 - theta2 < -F_PI/2) {
    return theta1 - theta2 + F_PI;
  } else if (theta1 - theta2 > F_PI/2) {
    return theta1 - theta2 - F_PI;
  }
  return theta1 - theta2;
}

void get_segments(vector<cv::Vec4f>& lines, vector<Segment>& segments, vector<int>& labels, int labels_num) {
  vector<float> theta_avgs(labels_num, 0), theta_refs(labels_num);
  vector<cv::Point2f> point_mus(labels_num, cv::Point2f(0, 0));
  vector<int> num(labels_num, 0);

  for (int i = 0; i < lines.size(); i++) {
    int label = labels[i];
    cv::Vec4f& line = lines[i];
    num[label]++;

    point_mus[label] += (cv::Point2f(line[0], line[1]) + cv::Point2f(line[2], line[3])) / 2.0;

    float theta, theta_delta;
    theta = atanf((line[1] - line[3]) / (line[0] - line[2] + EPSILON));

    if (num[label] == 1) {  // 当該ラベルの１つ目の線分
      theta_refs[label] = theta;
      continue;
    }
    theta_delta = angle_sub(theta, theta_refs[label]);
    theta_avgs[label] += theta_delta;
  }

  for (int l = 0; l < labels_num; l++) {
    theta_avgs[l] = theta_refs[l] + theta_avgs[l] / num[l];
    if (theta_avgs[l] > F_PI/2) theta_avgs[l] -= F_PI;
    else if (theta_avgs[l] < -F_PI/2) theta_avgs[l] += F_PI;
    point_mus[l] /= num[l];
  }

  // 各segmentにおいて、実際にlineがある(延長していない)範囲を求める
  vector<float> upper_dists(labels_num, -1);
  vector<float> lower_dists(labels_num, 1);
  for (int i = 0; i < lines.size(); i++) {
    int label = labels[i];
    cv::Vec4f& line = lines[i];
    float theta = theta_avgs[label];
    for (int j = 0; j <= 2; j += 2) {
      float dist = (cv::Point2f(line[j], line[j+1]) - point_mus[label])
                        .dot(cv::Point2f(cosf(theta), sinf(theta)));
      if (upper_dists[label] < dist) {
        upper_dists[label] = dist;
      } else if (lower_dists[label] > dist) {
        lower_dists[label] = dist;
      }
    }
  }

  for (int l = 0; l < labels_num; l++) {
    segments.push_back(Segment(point_mus[l], upper_dists[l], lower_dists[l], theta_avgs[l]));
  }
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
  cv::RNG rng(1234);
  cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
  vector<cv::Scalar> colors;
  for (int i = 0; i < segments.size(); i++) {
    Segment& seg = segments[i];
    cv::Scalar color(rng.uniform(0, 127), rng.uniform(0, 127), rng.uniform(0, 127));
    colors.push_back(color);
    cv::line(dst, seg.m_pe1, seg.m_p1, color, 2);
    cv::line(dst, seg.m_p1, seg.m_p2, color*2, 2);
    cv::line(dst, seg.m_p2, seg.m_pe2, color, 2);
  }
  for (int i = 0; i < segments.size(); i++) {
    Segment& seg = segments[i];
    cv::putText(dst, to_string(seg.m_id), seg.m_pm, cv::FONT_HERSHEY_PLAIN, 1.0, colors[i]*2);
  }
}

void draw_labeled_lines(cv::Mat& src, cv::Mat& dst, vector<cv::Vec4f>& lines, vector<int> labels, int labels_num) {
  cv::RNG rng(12341);
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
  // theta_delta = fabs(angle_sub(theta1, theta2));
  if (theta1 - theta2 < F_PI / 2) {
    theta_delta = theta1 - theta2;
    theta_avg = (theta1 + theta2) / 2;
  } else {
    theta_delta = theta2 - theta1 + F_PI;
    theta_avg = (theta1 + theta2 + F_PI) / 2;
    if (theta_avg > F_PI / 2) theta_avg -= F_PI;
  }
  // cout << "theta1 = " << theta1*180/F_PI << endl;
  // cout << "theta2 = " << theta2*180/F_PI << endl;
  // cout << "theta_delta = " << theta_delta*180/F_PI << endl;
  // cout << "theta_avg = " << theta_avg*180/F_PI << endl;

  // 向きが異なる2線分はマージしない
  if (theta_delta * 180 > LINE_EQUAL_DEGREE * F_PI) {
    return false;
  }

  float dist_point = min({powf(l1[0] - l2[0], 2) + powf(l1[1] - l2[1], 2),
                              powf(l1[0] - l2[2], 2) + powf(l1[1] - l2[3], 2),
                              powf(l1[2] - l2[0], 2) + powf(l1[3] - l2[1], 2),
                              powf(l1[2] - l2[2], 2) + powf(l1[3] - l2[3], 2)});
  dist_point = sqrtf(dist_point);
  float mx1 = (l1[0] + l1[2]) / 2;
  float my1 = (l1[1] + l1[3]) / 2;
  float mx2 = (l2[0] + l2[2]) / 2;
  float my2 = (l2[1] + l2[3]) / 2;
  float dist_ver = fabs(sinf(theta_avg) * (mx1 - mx2) - cosf(theta_avg) * (my1 - my2));

  // cout << "dist_ver = " << dist_ver << endl;
  // cout << "dist_point = " << dist_point << endl;

  // 端点が遠く、垂直距離が長い２線分はマージしない
  if (dist_point > POINT_EQUAL_DISTANCE && dist_ver > LINE_EQUAL_DISTANCE) {
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