#ifndef DETECT_H
#define DETECT_H

class Segment;
class Intersection;

class Segment {
private:
  enum Direction {
    HORIZONTAL,
    VERTICAL
  };
  float m_length;
  enum Direction m_type;
public:
  float m_theta;
	int m_id;
  cv::Point2f m_p1, m_p2, m_pm, m_pe1, m_pe2, m_dir;

  Segment(cv::Point2f& point_mu, float udist, float ldist, float theta);
  bool is_horizontal () const { return m_type == HORIZONTAL; }
  bool is_vertical() const { return m_type == VERTICAL; }

	static int num;
};

class Intersection{
private:
	enum Horizontal {
		LEFT,
		RIGHT
	};
	enum Vertical {
		TOP,
		BOTTOM
	};
	int m_seg_hor_id, m_seg_ver_id;
	float m_seglen_hor, m_seglen_ver;
	float m_segdist_hor, m_segdist_ver;
	float m_cross_degree;
	float m_segdeg_hor, m_segdeg_ver;
	cv::Point2f m_cross_point;
	enum Horizontal m_pos_hor;
	enum Vertical m_pos_ver;
	float m_score;

	void set_score();
public:
	Intersection(Segment& seg_hor, Segment& seg_ver, cv::Point2f& cross_point,
									float cross_degree, enum Horizontal pos_hor, enum Vertical pos_ver);
	static void get_intersections(std::vector<Segment>& segments, std::vector<Intersection>& intersections);
	float get_score() const { return m_score; }
	cv::Point2f get_cross_point() { return m_cross_point; }
	bool is_top() const { return m_pos_ver == TOP; }
	bool is_bottom() const { return m_pos_ver == BOTTOM; }
	bool is_left() const { return m_pos_hor == LEFT; }
	bool is_right() const { return m_pos_hor == RIGHT; }
	char m_description[300];	// For debug

	int m_id;
	static int num;
};

void init(cv::Mat& image);
float angle_sub(float theta1, float theta2);
void get_segments(std::vector<cv::Vec4f>& lines, std::vector<Segment>& segments, std::vector<int>& labels, int labels_num);
void remove_central_lines(std::vector<cv::Vec4f>& lines);
void remove_central_segments(std::vector<Segment>& segments);
void draw_lines(cv::Mat& src, cv::Mat& dst, std::vector<Segment>& segments);
void draw_labeled_lines(cv::Mat& src, cv::Mat& dst, std::vector<cv::Vec4f>& lines, std::vector<int> labels, int labels_num);
bool line_equal(const cv::Vec4f& l1, const cv::Vec4f& l2);
void get_combi_indice(int am, int bm, int cm, int dm, std::vector<std::vector<int> >& indice);


extern float LINE_EQUAL_DEGREE;         // 同じ線分とみなす線分間の最大角度
extern float LINE_EQUAL_DISTANCE;      // 同じ線分とみなす中点同士の最大垂直距離
// extern float POINT_EQUAL_DISTANCE;     // 別の線分の端点を同じ点とみなす最大距離
extern float LINE_INCLUDE_DISTANCE;    // 線分に点が含まれるとみなす最大距離
extern float LINE_CROSS_DEGREE;         // 直交とみなす最小角度
extern float CENTER_WIDTH;               // 画像中心部の幅
extern float CENTER_HEIGHT;              // 画像中心部の高さ

extern int POINT_IN_SECTION;						 // 画像の4箇所それぞれが含む候補点の数

extern const float EPSILON;  // 1e-7
extern const float F_PI;
extern const float SQRT2;
extern const float INF;

extern cv::Size img_size;

#endif