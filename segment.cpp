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
}
