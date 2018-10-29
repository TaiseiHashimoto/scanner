#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <cmath>
#include <cassert>
#include <utility>

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


float angle_sub(float theta1, float theta2) {
  if (theta1 - theta2 < -F_PI/2) {
    return theta1 - theta2 + F_PI;
  } else if (theta1 - theta2 > F_PI/2) {
    return theta1 - theta2 - F_PI;
  }
  return theta1 - theta2;
}

class Segment {
private:
  int m_type;
  float m_length;
  enum {
    HORIZONTAL,
    VERTICAL
  };
public:
  float m_theta;
  cv::Point2f m_p1, m_p2, m_pm, m_pe1, m_pe2, m_dir;

  Segment(cv::Point2f& point_mu, float udist, float ldist, float theta) {
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

  int intersection(Segment& seg, cv::Point2f& cpoint) {
    // if (m_type != seg.m_type && 
    //     fabs(angle_sub(m_theta, seg.m_theta)) * 180 < LINE_CROSS_DEGREE * F_PI) {
    //   cout << m_theta*180/F_PI << " " << seg.m_theta*180/F_PI << " " << 
    //       angle_sub(m_theta, seg.m_theta)*180/F_PI << endl;
    // }
    if (m_type == seg.m_type ||
        fabs(angle_sub(m_theta, seg.m_theta)) * 180 < LINE_CROSS_DEGREE * F_PI) {
      return -1;
    }

    //cout << m_pe1 << " " << m_pe2 << " ; " << seg.m_pe1 << " " << seg.m_pe2 << " ;; ";

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
      //cout << "-1" << endl;
      return -1;
    }

    // if (cpoint.x < seg_hor.m_p1.x) {
    //   if (cpoint.y < seg_ver.m_p1.y) cout << "0" << endl;
    //   else cout << "1" << endl;
    // } else {
    //   if (cpoint.y < seg_ver.m_p1.y) cout << "2" << endl;
    //   else cout << "3" << endl;
    // }

    if (cpoint.x < seg_hor.m_p1.x) {
      return cpoint.y < seg_ver.m_p1.y ? 0 : 1;
    } else {
      return cpoint.y < seg_ver.m_p1.y ? 2 : 3;
    }
  }

  bool isHorizontal() {
    return m_type == HORIZONTAL;
  }

  bool isVertical() {
    return m_type == VERTICAL;
  }
};

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
      assert(theta_refs[label] < F_PI/2 || theta_refs[label] > -F_PI/2);
      continue;
    }
    theta_delta = angle_sub(theta, theta_refs[label]);
    // 角度の差が2倍に収まっている(角度計算の誤り検出)
    assert(theta_delta * 180 < LINE_EQUAL_DEGREE * F_PI * 2);
    theta_avgs[label] += theta_delta;
  }

  for (int l = 0; l < labels_num; l++) {
    theta_avgs[l] = theta_refs[l] + theta_avgs[l] / num[l];
    if (theta_avgs[l] > F_PI/2) theta_avgs[l] -= F_PI;
    else if (theta_avgs[l] < -F_PI/2) theta_avgs[l] += F_PI;
    point_mus[l] /= num[l];
  }

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
  for (int i = 0; i < segments.size(); i++) {
    Segment& seg = segments[i];
    cv::Scalar color(rng.uniform(0, 127), rng.uniform(0, 127), rng.uniform(0, 127));
    cv::line(dst, seg.m_pe1, seg.m_p1, color, 2);
    cv::line(dst, seg.m_p1, seg.m_p2, color*2, 2);
    cv::line(dst, seg.m_p2, seg.m_pe2, color, 2);
  }
}

void draw_labeled_lines(cv::Mat& src, cv::Mat& dst, vector<cv::Vec4f>& lines, vector<int> labels, int labels_num) {
  cv::RNG rng(12345);
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
  float theta1, theta2, theta_delta, theta_avg, theta_orth;
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
  if (theta_avg > 0) {
    theta_orth = theta_avg - F_PI / 2;
  } else {
    theta_orth = theta_avg + F_PI / 2;
  }

  if (theta_delta * 180 > LINE_EQUAL_DEGREE * F_PI) {
    return false;
  }

  float dist2_point = min({powf(l1[0] - l2[0], 2) + powf(l1[1] - l2[1], 2),
                          powf(l1[0] - l2[2], 2) + powf(l1[1] - l2[3], 2),
                          powf(l1[2] - l2[0], 2) + powf(l1[3] - l2[1], 2),
                          powf(l1[2] - l2[2], 2) + powf(l1[3] - l2[3], 2)});
  float mx1 = (l1[0] + l1[2]) / 2;
  float my1 = (l1[1] + l1[3]) / 2;
  float mx2 = (l2[0] + l2[2]) / 2;
  float my2 = (l2[1] + l2[3]) / 2;
  float dist_vert = fabs(cosf(theta_orth) * (mx1 - mx2) + sinf(theta_orth) * (my1 - my2));
  if (dist2_point > POINT_EQUAL_DISTANCE && dist_vert > LINE_EQUAL_DISTANCE) {
    return false;
  }
  return true;
}

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

  // 各種値を初期化
  img_size = image.size();
  cout << img_size << endl;
  int len_avg = (img_size.width + img_size.height) / 2;
  LINE_EQUAL_DEGREE = 5;
  LINE_EQUAL_DISTANCE = len_avg * 0.003;
  POINT_EQUAL_DISTANCE = len_avg * 0.02;
  LINE_INCLUDE_DISTANCE = len_avg * 0.006;
  LINE_CROSS_DEGREE = 60;
  CENTER_WIDTH = img_size.width * 0.5;
  CENTER_HEIGHT = img_size.height * 0.5;

  cout << "LINE_EQUAL_DEGREE" << LINE_EQUAL_DEGREE << endl;
  cout << "LINE_EQUAL_DISTANCE" << LINE_EQUAL_DISTANCE << endl;
  cout << "POINT_EQUAL_DISTANCE" << POINT_EQUAL_DISTANCE << endl;
  cout << "LINE_INCLUDE_DISTANCE" << LINE_INCLUDE_DISTANCE << endl;
  cout << "LINE_CROSS_DEGREE" << LINE_CROSS_DEGREE << endl;
  cout << "CENTER_WIDTH" << CENTER_WIDTH << endl;
  cout << "CENTER_HEIGHT" << CENTER_HEIGHT << endl;
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
