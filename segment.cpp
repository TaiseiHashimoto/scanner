#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include "detect.hpp"

Segment::Segment(cv::Point2f& point_mu, float udist, float ldist, float theta) {
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

int Segment::intersection(Segment& seg, cv::Point2f& cpoint) {
  if (m_type == seg.m_type ||
      fabs(angle_sub(m_theta, seg.m_theta)) * 180 < LINE_CROSS_DEGREE * F_PI) {
    return -1;
  }

  cv::Point2f x = seg.m_pe1 - m_pe1;
  cv::Point2f d1 = m_pe2 - m_pe1;
  cv::Point2f d2 = seg.m_pe2 - seg.m_pe1;
  float t1 = (x.x * d2.y - x.y * d2.x) / (d1.x * d2.y - d1.y * d2.x);
  cpoint = m_pe1 + d1 * t1;

  Segment seg_hor = *this, seg_ver = seg;
  if (m_type == VERTICAL) swap(seg_hor, seg_ver);

  if ((cpoint.x > seg_hor.m_p1.x + LINE_INCLUDE_DISTANCE &&
        cpoint.x < seg_hor.m_p2.x - LINE_INCLUDE_DISTANCE) ||
      (cpoint.y > seg_ver.m_p1.y + LINE_INCLUDE_DISTANCE &&
        cpoint.y < seg_ver.m_p2.y - LINE_INCLUDE_DISTANCE)) {
    return -1;
  }

  if (cpoint.x < seg_hor.m_p1.x) {
    return cpoint.y < seg_ver.m_p1.y ? 0 : 1;
  } else {
    return cpoint.y < seg_ver.m_p1.y ? 2 : 3;
  }
}

bool Segment::isHorizontal() {
  return m_type == HORIZONTAL;
}

bool Segment::isVertical() {
  return m_type == VERTICAL;
}
