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

def scan(main_color, intermediates=None):
    main_gray = utility.shrink_img(main_color)    # 処理速度を上げるため縮小する
    main_gray = cv2.cvtColor(main_gray, cv2.COLOR_BGR2GRAY)
    size_color = main_color.shape[:2]
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
    if intermediates is not None:
        intermediates['lines'] = utility.draw_lines(main_gray, lines, labels, labels_num)

    segments = Segments(lines, labels, labels_num, size_gray)
    print(f"Num of segments: {segments.num} => ", end="")
    segments.remove_central(size_gray)
    print(segments.num)
    if intermediates is not None:
        intermediates['segments'] = utility.draw_segments(main_gray, segments)

    intersections = Intersections(segments, size_gray)
    print(f"Num of intersections: {intersections.num}")
    if intermediates is not None:
        intermediates['intersections'] = utility.draw_intersections(main_gray, intersections)
    
    df = ml_model.prepare_data(intersections, size_gray)
    scores = ml_model.get_score(df)
    indice = np.argsort(scores)[::-1]

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
    if len(vertex_lt) == 0:
        print("no vertex at left top was found")
        return None
    if len(vertex_rt) == 0:
        print("no vertex at right top was found")
        return None
    if len(vertex_lb) == 0:
        print("no vertex at left bottom was found")
        return None
    if len(vertex_rb) == 0:
        print("no vertex at right bottom was found")
        return None

    idx_lt, idx_rt, idx_lb, idx_rb = utility.get_best_set(intersections, scores, vertex_lt, vertex_rt, vertex_lb, vertex_rb)

    print(f"selected vertexes: ({idx_lt}, {idx_rt}, {idx_lb}, {idx_rb})")
    if intermediates is not None:
        intermediates['detected'] = utility.draw_detected(main_gray, intersections, idx_lt, idx_rt, idx_lb, idx_rb)

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

    # M_gray = cv2.getPerspectiveTransform(src_points_gray, dst_points_gray)
    M_color = cv2.getPerspectiveTransform(src_points_color, dst_points_color)
    # main_gray = cv2.warpPerspective(main_gray, M_gray, size_gray[::-1])
    main_color = cv2.warpPerspective(main_color, M_color, size_color[::-1])
    return main_color


def blend(main_color, sub_colors, intermediates=None):
    if len(sub_colors) == 0:
        print("No sub image available")
        return None

    main_gray = utility.shrink_img(main_color)    # 処理速度を上げるため縮小する
    main_gray = cv2.cvtColor(main_gray, cv2.COLOR_BGR2GRAY)

    sub_shrinkeds = match.align(main_gray, sub_colors)
    if intermediates is not None:
        for i in range(len(sub_shrinkeds)):
            intermediates[f"sub_aligned_{i}"] = sub_shrinkeds[i].copy()

    if len(sub_shrinkeds) == 0:
        print("Unable to blend (bad matching)")
        return main_color

    size_color = main_color.shape[:2]
    size_gray = main_gray.shape
    main_shrinked = utility.shrink_img(main_color)
    
    sub_grays = []
    sub_colors = []
    for sub_shrinked in sub_shrinkeds:
        sub_color = cv2.resize(sub_shrinked, size_color[::-1])
        sub_colors.append(sub_color)
        sub_grays.append(cv2.cvtColor(sub_shrinked, cv2.COLOR_BGR2GRAY))

    # state: 0 (完全に白飛び) ~ 1 (全く白飛びしていない)
    main_state = 1 - match.detect_overexposed(main_shrinked)
    if intermediates is not None:
        intermediates['main_state'] = utility.draw_score(main_shrinked, main_state)

    sub_states = []
    for i in range(len(sub_grays)):
        state = 1 - match.detect_overexposed(sub_shrinkeds[i])
        state[sub_grays[i] == 0] = 0   # 見切れている部分(真っ黒)は使わない
        sub_states.append(state)
        if intermediates is not None:
            intermediates[f"sub_state_{i}"] = utility.draw_score(sub_shrinkeds[i], state)
    sub_states = np.array(sub_states, dtype=np.float32)

    sub_states_max = np.clip(np.max(sub_states, axis=0), 1e-6, 1)
    main_weight = np.clip((np.tanh((main_state - 0.93) * 15) + 1.0) * 0.5 + (np.tanh((main_state/sub_states_max - 1.0) * 1.0) + 1.0) * 0.3, 1e-6, 1)

    sub_weights = []
    for state in sub_states:
        sub_weights.append(np.exp((state - 0.95)*5))
    sub_weights = np.array(sub_weights)
    sub_weights_sum = np.sum(sub_weights, axis=0)
    sub_weights[:, sub_weights_sum > 0] /= sub_weights_sum[sub_weights_sum > 0]
    sub_weights *= (1 - main_weight)

    # 初めに小さい画像でブレンドして、明度情報を求めておく
    bld = main_shrinked * main_weight[:, :, np.newaxis]
    for img, weight in zip(sub_shrinkeds, sub_weights):
        bld += img * weight[:, :, np.newaxis]
    bld = np.clip(bld, 0, 255)

    if intermediates is not None:
        intermediates['main_weight'] = utility.draw_score(main_shrinked, main_weight)
        for i in range(len(sub_weights)):
            intermediates[f"sub_weight_{i}"] = utility.draw_score(sub_shrinkeds[i], sub_weights[i])
        intermediates['blended_dark'] = bld.astype(np.uint8)

    # HSV画像を用意(明度情報を使う)
    bld_hsv = cv2.cvtColor(bld, cv2.COLOR_BGR2HSV)
    sub_vs = []
    for img in sub_shrinkeds:
        sub_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sub_vs.append(sub_hsv[:, :, 2])

    # 暗い領域を明るくするため、明るくしてよい余裕を求める
    bright_margin = np.zeros(size_gray, dtype=np.float32)
    for img_v, state in zip(sub_vs, sub_states):
        bright_margin = np.maximum((img_v - bld_hsv[:, :, 2]) * (np.tanh((state - 0.99) * 15) + 1.0) * 0.3, bright_margin)

    # ぼかしをかけることで境目を目立たなくさせる
    bright_margin = cv2.blur(bright_margin, (20, 20))
    if intermediates is not None:
        intermediates['bright_margin'] = utility.draw_score(main_shrinked, bright_margin)

    # もとの画像サイズでブレンドする
    main_weight = cv2.resize(main_weight, size_color[::-1])
    blended = main_color * main_weight[:, :, np.newaxis]
    sub_shrinkeds = np.array(sub_shrinkeds, dtype=np.float32)
    blend_img = np.sum(sub_shrinkeds * sub_weights[:, :, :, np.newaxis], axis=0)
    blended = np.clip(blended + blend_img, 0, 255)

    bright_margin = cv2.resize(bright_margin, size_color[::-1])
    blended_hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
    blended_hsv[:, :, 2] = np.clip(blended_hsv[:, :, 2] + bright_margin, 0, 255)
    blended = cv2.cvtColor(blended_hsv, cv2.COLOR_HSV2BGR)

    return blended.astype(np.uint8)


if __name__ == '__main__':
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

    intermediates = {}
    main_color = scan(main_color, intermediates)
    if not main_color:
        print("scan failed")
        exit(1)

    cv2.imshow("warped_main", main_color)
    cv2.imwrite("warped_main.jpeg", main_color)
    # cv2.imshow("warped_main_gray", main_gray)
    # cv2.waitKey()

    sub_shrinkeds = []
    for filename in subfilenames:
        sub_color = cv2.imread(filename)
        assert sub_color is not None, f"Cannot read {filename}"
        sub_shrinked = utility.shrink_img(sub_color)
        sub_shrinkeds.append(sub_shrinked)
    
    blended = blend(main_color, sub_shrinkeds, intermediates)
    if not blended:
        print('blend failed')
        exit(1)

    cv2.imshow("blended", blended)
    cv2.imwrite("blended.jpeg", blended)

    duration_ms = (cv2.getTickCount() - start) * 1000 / cv2.getTickFrequency()
    print(f"It took {duration_ms} ms.")

    cv2.waitKey()
