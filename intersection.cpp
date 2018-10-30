#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <numeric>
#include "detect.hpp"

using namespace std;

static int ninters = 0;	// for debug

int Intersection::num;

Intersection::Intersection(Segment& seg_hor, Segment& seg_ver, cv::Point2f& cross_point,
									float cross_degree, enum Horizontal pos_hor, enum Vertical pos_ver) {
  m_id = num++;
  m_seg_hor_id = seg_hor.m_id;
  m_seg_ver_id = seg_ver.m_id;
	m_seglen_hor = cv::norm(seg_hor.m_p1 - seg_hor.m_p2);
	m_seglen_ver = cv::norm(seg_ver.m_p1 - seg_ver.m_p2);

	m_cross_point = cross_point;
	m_cross_degree = cross_degree;
	m_pos_hor = pos_hor;
	m_pos_ver = pos_ver;
	m_segdeg_hor = seg_hor.m_theta;
	m_segdeg_ver = seg_ver.m_theta;

	if (m_pos_hor == LEFT)
		m_segdist_hor = cv::norm(seg_hor.m_p1 - cross_point);
	else if (m_pos_hor == RIGHT)
		m_segdist_hor = cv::norm(seg_hor.m_p2 - cross_point);
	if (m_pos_ver == TOP)
		m_segdist_ver = cv::norm(seg_ver.m_p1 - cross_point);
	else if (m_pos_ver == BOTTOM)
		m_segdist_ver = cv::norm(seg_ver.m_p2 - cross_point);

	set_score();
}

void Intersection::get_intersections(vector<Segment>& segments, vector<Intersection>& intersections) {
	for (int i = 0; i < segments.size(); i++) {
		for (int j = i + 1; j < segments.size(); j++) {
			if ((segments[i].is_horizontal() && segments[j].is_horizontal()) ||
					(segments[i].is_vertical() && segments[j].is_vertical()))
				continue;
			Segment seg_hor = segments[i], seg_ver = segments[j];
			if (seg_hor.is_vertical()) swap(seg_hor, seg_ver);

			float cross_degree = fabs(angle_sub(seg_hor.m_theta, seg_ver.m_theta));
		  if (cross_degree * 180 < LINE_CROSS_DEGREE * F_PI) continue;

		  cv::Point2f x = seg_ver.m_pe1 - seg_hor.m_pe1;
		  cv::Point2f d1 = seg_hor.m_pe2 - seg_hor.m_pe1;
		  cv::Point2f d2 = seg_ver.m_pe2 - seg_ver.m_pe1;
		  float t1 = (x.x * d2.y - x.y * d2.x) / (d1.x * d2.y - d1.y * d2.x);
		  cv::Point2f cross_point = seg_hor.m_pe1 + d1 * t1;

		  enum Horizontal pos_hor;
		  enum Vertical pos_ver;

		  if (cross_point.x < seg_hor.m_p1.x + LINE_INCLUDE_DISTANCE) {
		  	if (cross_point.x > img_size.width / 2 - CENTER_WIDTH / 2) continue;	
		  	pos_hor = LEFT;
		  } else if (cross_point.x > seg_hor.m_p2.x - LINE_INCLUDE_DISTANCE) {
				if (cross_point.x < img_size.width / 2 + CENTER_WIDTH / 2) continue;
				pos_hor = RIGHT;
		  } else continue;
	  	
	  	if (cross_point.y < seg_ver.m_p1.y + LINE_INCLUDE_DISTANCE) {
		  	if (cross_point.y > img_size.height / 2 - CENTER_HEIGHT / 2) continue;
		  	pos_ver = TOP; 
	  	} else if (cross_point.y > seg_ver.m_p2.y - LINE_INCLUDE_DISTANCE) {
	  		if (cross_point.y < img_size.height / 2 + CENTER_HEIGHT / 2) continue;
	  		pos_ver = BOTTOM;
	  	} else continue;

		  intersections.push_back(Intersection(seg_hor, seg_ver, cross_point,
		  																							cross_degree, pos_hor, pos_ver));
		}
	}
}

void Intersection::set_score() {
	const float w_seglen = 0.03;
	const float w_segdist = 1;//500;
	const float w_degree = 1000;
	const float w_position = 0.002;
	const float score_clip = 20.0;		// 大きすぎるスコアは頭打ちにする

	// printf("Num. %d\n", ninters++);

	m_score = 0;
	char pos_str[15];
	if (m_pos_hor == LEFT)
		sprintf(pos_str, "LEFT_");
	else
		sprintf(pos_str, "RIGHT_");
	if (m_pos_ver == TOP)
		sprintf(pos_str, "%sTOP", pos_str);
	else
		sprintf(pos_str, "%sBOTTOM", pos_str);

	sprintf(m_description, "# %d  segment: %d, %d  position: %s\n", m_id, m_seg_hor_id, m_seg_ver_id, pos_str);

	float score_seglen = m_seglen_hor*m_seglen_ver/40 + m_seglen_hor + m_seglen_ver;
	score_seglen *= w_seglen;
	m_score += min(score_seglen, score_clip);
	sprintf(m_description, "%sseglen: %f  (%f, %f)\n", m_description, score_seglen, m_seglen_hor, m_seglen_ver);

	// float score_segdist = 1.0 / (m_segdist_hor + m_segdist_ver + 1);
	float score_segdist = -sqrtf(powf(m_segdist_hor, 2) + powf(m_segdist_ver, 2));
	score_segdist *= w_segdist;
	m_score += max(score_segdist, -score_clip);
	sprintf(m_description, "%ssegdist: %f  (%f, %f)\n", m_description, score_segdist, m_segdist_hor, m_segdist_ver);

	float score_cross_degree = -powf(m_cross_degree - F_PI/2, 2)
																	-powf(angle_sub(m_segdeg_hor, 0), 2) / 2
																	-powf(angle_sub(m_segdeg_ver, F_PI/2), 2) / 2;
	score_cross_degree *= w_degree;
	m_score += max(score_cross_degree, -score_clip);
	sprintf(m_description, "%scross_degree: %f  (%f, %f, %f)\n", m_description, score_cross_degree,
							m_cross_degree*180/F_PI, m_segdeg_hor*180/F_PI, m_segdeg_ver*180/F_PI);

	float score_position = 0;
	if (m_pos_hor == LEFT)
		score_position -= powf(m_cross_point.x, 2);
	else if (m_pos_hor == RIGHT)
		score_position -= powf((img_size.width - m_cross_point.x), 2);
	if (m_pos_ver == TOP)
		score_position -= powf(m_cross_point.y, 2);
	else if (m_pos_ver == BOTTOM)
		score_position -= powf(img_size.height - m_cross_point.y, 2);
	// score_position = -sqrtf(score_position);
	score_position *= w_position;
	m_score += max(score_position, -score_clip);
	sprintf(m_description, "%sposition: %f  [%f, %f]\n", m_description, score_position, m_cross_point.x, m_cross_point.y);

	sprintf(m_description, "%sscore... %f\n\n", m_description, m_score);
}
