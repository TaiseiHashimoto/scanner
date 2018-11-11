import cv2
import cv2.ximgproc
import numpy as np
import sys
import os
import glob
import utility
import partition
import ml_model
import match
from lines import Lines
from segments import Segments
from intersections import Intersections


def main():
    start = cv2.getTickCount()
    foldername = os.path.join("data", sys.argv[1])
    mainfilename = os.path.join(foldername, "main.jpeg")
    subfilenames = glob.glob(foldername + "/sub_[0-9].jpeg")
    datafilename = os.path.join(foldername, "data.csv")

    img_color = cv2.imread(mainfilename)
    assert img_color is not None, f"Cannot read {mainfilename}"
    main_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    size_color = np.array(img_color.shape)

    if size_color[0] > size_color[1] and size_color[0] > 600:
        main_gray = cv2.resize(main_gray, (600, 600*size_color[1]//size_color[0]))
    elif size_color[0] < size_color[1] and size_color[1] > 600:
        main_gray = cv2.resize(main_gray, (600*size_color[0]//size_color[1]), 600)
    size_gray = np.array(main_gray.shape)

    length_threshold = 20
    distance_threshold = 1.4142
    canny_th1 = 5.0
    canny_th2 = 50.0
    canny_size = 3
    do_merge = False
    do_merge = False
    fld = cv2.ximgproc.createFastLineDetector(length_threshold, distance_threshold, canny_th1, canny_th2, canny_size, do_merge)
    line_pnts = fld.detect(main_gray)
    line_pnts = np.array(line_pnts).reshape((-1, 4))
    lines = Lines(line_pnts)
    print(f"Num of lines: {lines.num} => ", end="")
    lines.remove_central(size_gray)
    print(lines.num)

    equal = lines.equal()
    labels = partition.partition(lines.num, equal)
    labels_num = len(np.unique(labels))
    tmp = utility.draw_lines(main_gray, lines, labels, labels_num)
    cv2.imshow("lines detected", tmp)

    segments = Segments(lines, labels, labels_num, size_gray)
    print(f"Num of segments: {segments.num} => ", end="")
    segments.remove_central(size_gray)
    print(segments.num)

    tmp = utility.draw_segments(main_gray, segments)
    cv2.imshow("setmgents detected", tmp)

    intersections = Intersections(segments, size_gray)
    print(f"Num of intersections: {intersections.num}")
    
    data = ml_model.prepare_data(intersections, size_gray)
    scores = ml_model.get_score(data)
    np.savetxt(datafilename, data, delimiter=',')
    indice = np.argsort(scores)[::-1]

    tmp = utility.draw_intersections(main_gray, intersections, indice)
    cv2.imshow("intersections detected", tmp)

    points_per_section = 1
    vertex_lt = []
    vertex_rt = []
    vertex_lb = []
    vertex_rb = []
    for idx in indice:
        if intersections.is_left[idx] and intersections.is_top[idx]:
            vertex_lt.append(idx)
        elif intersections.is_right[idx] and intersections.is_top[idx]:
            vertex_rt.append(idx)
        elif intersections.is_left[idx] and intersections.is_bottom[idx]:
            vertex_lb.append(idx)
        elif intersections.is_right[idx] and intersections.is_bottom[idx]:
            vertex_rb.append(idx)

        if len(vertex_lt) >= points_per_section and \
            len(vertex_rt) >= points_per_section and \
            len(vertex_lb) >= points_per_section and \
            len(vertex_rb) >= points_per_section:
            break

    assert len(vertex_lt) > 0, "no vertex at left top was found"
    assert len(vertex_rt) > 0, "no vertex at right top was found"
    assert len(vertex_lb) > 0, "no vertex at left bottom was found"
    assert len(vertex_rb) > 0, "no vertex at right bottom was found"
    idx_lt = vertex_lt[0]
    idx_rt = vertex_rt[0]
    idx_lb = vertex_lb[0]
    idx_rb = vertex_rb[0]

    tmp = utility.draw_detected(main_gray, intersections, idx_lt, idx_rt, idx_lb, idx_rb)
    cv2.imshow("detected", tmp)

    scale = np.array(img_color.shape[:2]) / size_gray
    scale = scale[::-1]    # 縦横 => 横縦
    src_points = np.array([ \
        intersections.cross_pnt[idx_lt] * scale, \
        intersections.cross_pnt[idx_rt] * scale, \
        intersections.cross_pnt[idx_lb] * scale, \
        intersections.cross_pnt[idx_rb] * scale
    ], dtype=np.float32)
    dst_points = np.array([ \
        (0, 0),
        (img_color.shape[1], 0),
        (0, img_color.shape[0]),
        (img_color.shape[1], img_color.shape[0])
    ], dtype=np.float32)

    trans_mat = cv2.getPerspectiveTransform(src_points, dst_points)
    img_color = cv2.warpPerspective(img_color, trans_mat, (img_color.shape[:2][::-1]))

    cv2.imshow("warped", img_color)

    if len(subfilenames) > 0:
        sub_colors = []
        for filename in subfilenames:
            sub_color = cv2.imread(filename)
            assert sub_color is not None, f"Cannot read {filename}"
            sub_colors.append(sub_color)
        sub_colors, matched_pnts = match.homography(img_color, sub_colors)
        blended = match.blend(img_color, sub_colors, matched_pnts)
        cv2.imshow("blended", blended)
        cv2.imwrite("blended.jpeg", blended)

    duration_ms = (cv2.getTickCount() - start) * 1000 / cv2.getTickFrequency()
    print(f"It took {duration_ms} ms.")

    cv2.waitKey()

if __name__ == '__main__':
    main()