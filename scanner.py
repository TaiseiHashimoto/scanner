import cv2
import cv2.ximgproc
import numpy as np
import sys
import os
import glob
import partition
import utility
import match
import ml_model
from lines import Lines
from segments import Segments
from intersections import Intersections
from pathlib import Path

def main():
    start = cv2.getTickCount()
    foldername = os.path.join("data", sys.argv[1])
    if len(sys.argv) == 2:
        mainfilename = os.path.join(foldername, "main.jpeg")
        subfilenames = glob.glob(foldername + "/sub_[0-9].jpeg")
        datafilename = os.path.join(foldername, "data.csv")
    elif sys.argv[2] == 'big':
        mainfilename = os.path.join(foldername, "main_big.jpeg")
        subfilenames = glob.glob(foldername + "/sub_[0-9]_big.jpeg")
        datafilename = os.path.join(foldername, "data_big.csv")

    Path(os.path.join(foldername, 'ans.txt')).touch()

    main_color = cv2.imread(mainfilename)
    assert main_color is not None, f"Cannot read {mainfilename}"
    main_gray = cv2.cvtColor(main_color, cv2.COLOR_BGR2GRAY)
    size_color = main_color.shape[:2]
    main_gray = utility.shrink_img(main_gray)    # 処理速度を上げるため縮小する(最大600pixel)
    size_gray = main_gray.shape

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
    # cv2.imshow("lines detected", tmp)

    segments = Segments(lines, labels, labels_num, size_gray)
    print(f"Num of segments: {segments.num} => ", end="")
    segments.remove_central(size_gray)
    print(segments.num)

    tmp = utility.draw_segments(main_gray, segments)
    cv2.imshow("setmgents detected", tmp)

    intersections = Intersections(segments, size_gray)
    print(f"Num of intersections: {intersections.num}")
    
    df = ml_model.prepare_data(intersections, size_gray)
    scores = ml_model.get_score(df)
    df.to_csv(datafilename, index=None)
    indice = np.argsort(scores)[::-1]

    tmp = utility.draw_intersections(main_gray, intersections, indice)
    cv2.imshow("intersections detected", tmp)

    points_per_section = 3
    vertex_lt = []
    vertex_rt = []
    vertex_lb = []
    vertex_rb = []
    for idx in indice:
        if intersections.is_left[idx] and intersections.is_top[idx] and len(vertex_lt) < points_per_section:
            vertex_lt.append(idx)
        elif intersections.is_right[idx] and intersections.is_top[idx] and  len(vertex_rt) < points_per_section:
            vertex_rt.append(idx)
        elif intersections.is_left[idx] and intersections.is_bottom[idx] and len(vertex_lb) < points_per_section:
            vertex_lb.append(idx)
        elif intersections.is_right[idx] and intersections.is_bottom[idx] and  len(vertex_rb) < points_per_section:
            vertex_rb.append(idx)

        if len(vertex_lt) >= points_per_section and \
            len(vertex_rt) >= points_per_section and \
            len(vertex_lb) >= points_per_section and \
            len(vertex_rb) >= points_per_section:
            break

    # cv2.waitKey()
    assert len(vertex_lt) > 0, "no vertex at left top was found"
    assert len(vertex_rt) > 0, "no vertex at right top was found"
    assert len(vertex_lb) > 0, "no vertex at left bottom was found"
    assert len(vertex_rb) > 0, "no vertex at right bottom was found"

    idx_lt, idx_rt, idx_lb, idx_rb = utility.get_best_set(intersections, scores, vertex_lt, vertex_rt, vertex_lb, vertex_rb)

    print(f"selected vertexes: ({idx_lt}, {idx_rt}, {idx_lb}, {idx_rb})")

    tmp = utility.draw_detected(main_gray, intersections, idx_lt, idx_rt, idx_lb, idx_rb)
    cv2.imshow("detected", tmp)

    src_points_gray = np.array([ \
        intersections.cross_pnt[idx_lt], \
        intersections.cross_pnt[idx_rt], \
        intersections.cross_pnt[idx_lb], \
        intersections.cross_pnt[idx_rb]
    ], dtype=np.float32)
    dst_points_gray = np.array([ \
        (0, 0),
        (size_gray[1], 0),
        (0, size_gray[0]),
        (size_gray[1], size_gray[0])
    ], dtype=np.float32)
    scale = np.array(size_color, dtype=np.float32) / np.array(size_gray, dtype=np.float32)
    scale = scale[::-1]
    src_points_color = src_points_gray * scale
    dst_points_color = dst_points_gray * scale

    M_gray = cv2.getPerspectiveTransform(src_points_gray, dst_points_gray)
    M_color = cv2.getPerspectiveTransform(src_points_color, dst_points_color)
    main_gray = cv2.warpPerspective(main_gray, M_gray, size_gray[::-1])
    main_color = cv2.warpPerspective(main_color, M_color, size_color[::-1])

    cv2.imshow("warped_main", main_color)
    cv2.imwrite("warped_main.jpeg", main_color)
    # cv2.imshow("warped_main_gray", main_gray)
    # cv2.waitKey()

    if len(subfilenames) > 0:
        sub_colors = []
        for filename in subfilenames:
            sub_color = cv2.imread(filename)
            assert sub_color is not None, f"Cannot read {filename}"
            sub_colors.append(sub_color)
        sub_aligned = match.homography(main_color, main_gray, sub_colors)
        if len(sub_aligned) > 0:
            blended = match.blend(main_color, main_gray, sub_aligned)
        else:
            print("no blend (bad matching)")
            blended = main_color
        cv2.imshow("blended", blended)
        cv2.imwrite("blended.jpeg", blended)

    duration_ms = (cv2.getTickCount() - start) * 1000 / cv2.getTickFrequency()
    print(f"It took {duration_ms} ms.")

    cv2.waitKey()

if __name__ == '__main__':
    main()