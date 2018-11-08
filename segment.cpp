#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include "detect.hpp"

using namespace std;

int Segment::num;

Segment::Segment(cv::Point2f& point_mu, float udist, float ldist, float theta) {
  m_id = num++;
  m_pm = point_mu;
  m_theta = theta;

  m_dir = cv::Point2f(cosf(m_theta), sinf(m_theta));
  float ext_len = max(img_size.width, img_size.height) * SQRT2;
  // m_p1→m_p2 or m_p1↓p2の向き
  m_p1 = m_pm + m_dir * ldist;
  m_p2 = m_pm + m_dir * udist;
  m_pe1 = m_pm - m_dir * ext_len;
  m_pe2 = m_pm + m_dir * ext_len;
  m_length = cv::norm(m_p1 - m_p2);

  if (m_theta > -F_PI/4 && m_theta < F_PI/4) {
    m_type = HORIZONTAL;
  } else {
    m_type = VERTICAL;
    // m_p1↑p2の場合
    if (m_theta < 0) {
      swap(m_p1, m_p2);
      swap(m_pe1, m_pe2);
    }
  }

  // clipLineはcv::Pointを渡す必要がある
  cv::Point tmp1 = m_pe1, tmp2 = m_pe2;
  clipLine(img_size, tmp1, tmp2);
  m_pe1 = tmp1; m_pe2 = tmp2;

  // if (m_id == 30) {
  //   printf("id:%d theta=%f %d\n", m_id, m_theta*180/F_PI, m_type);
  // }
}

void Segment::get_segments(vector<cv::Vec4f>& lines, vector<Segment>& segments, vector<int>& labels, int labels_num) {
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