#ifndef DETECT_H
#define DETECT_H

using namespace std;

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

  Segment(cv::Point2f& point_mu, float udist, float ldist, float theta);
  int intersection(Segment& seg, cv::Point2f& cpoint);
  bool isHorizontal();
  bool isVertical();
};

void init(cv::Mat& image);
float angle_sub(float theta1, float theta2);
void get_segments(vector<cv::Vec4f>& lines, vector<Segment>& segments, vector<int>& labels, int labels_num);
void remove_central_segments(vector<Segment>& segments);
void draw_lines(cv::Mat& src, cv::Mat& dst, vector<Segment>& segments);
void draw_labeled_lines(cv::Mat& src, cv::Mat& dst, vector<cv::Vec4f>& lines, vector<int> labels, int labels_num);
bool line_equal(const cv::Vec4f& l1, const cv::Vec4f& l2);


extern float LINE_EQUAL_DEGREE;         // 同じ線分とみなす線分間の最大角度
extern float LINE_EQUAL_DISTANCE;      // 同じ線分とみなす中点同士の最大垂直距離
extern float POINT_EQUAL_DISTANCE;     // 別の線分の端点を同じ点とみなす最大距離
extern float LINE_INCLUDE_DISTANCE;    // 線分に点が含まれるとみなす最大距離
extern float LINE_CROSS_DEGREE;         // 直交とみなす最小角度
extern float CENTER_WIDTH;               // 画像中心部の幅
extern float CENTER_HEIGHT;              // 画像中心部の高さ

extern const float EPSILON;  // 1e-7
extern const float F_PI;
extern const float SQRT2;

extern cv::Size img_size;

#endif