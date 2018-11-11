import cv2
import numpy as np
import utility

def homography(main_color, main_gray, sub_colors):
    size_color = main_color.shape[:2]
    size_gray = main_gray.shape
    sub_grays = []
    for sub_color in sub_colors:
        shrinked = utility.shrink_img(sub_color)
        sub_grays.append(cv2.cvtColor(shrinked, cv2.COLOR_BGR2GRAY))

    akaze = cv2.AKAZE_create()
    main_kps, main_dscs = akaze.detectAndCompute(main_gray, None)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)

    for i in range(len(sub_colors)):
        kps, dscs = akaze.detectAndCompute(sub_grays[i], None)
        matches = bf.match(main_dscs, dscs)
        matches.sort(key=lambda m: m.distance)
        num_good_matches = min(int(len(matches) * 0.5), 100)
        if num_good_matches < 10:
            num_good_matches = len(matches)
        assert num_good_matches >= 4, f"Num of matches is too low. ({num_good_matches})"
        print(f"Num of matches: {len(matches)}")
        print(f"Num of good matches: {num_good_matches}")
        matches = matches[:num_good_matches]

        # tmp = cv2.drawMatches(main_img, main_kps, img, kps, matches, None)
        # cv2.imshow(f"matches_{i}", tmp)

        main_pnts = []
        pnts = []
        for j in range(num_good_matches):
            main_pnts.append(main_kps[matches[j].queryIdx].pt)
            pnts.append(kps[matches[j].trainIdx].pt)
        main_pnts = np.array(main_pnts)
        pnts = np.array(pnts)

        trans_mat = cv2.findHomography(pnts, main_pnts, cv2.RANSAC)[0]
        sub_colors[i] = cv2.warpPerspective(sub_colors[i], trans_mat, size_color[::-1])

    return sub_colors

def blend(main_color, main_gray, sub_colors):
    size_color = main_color.shape[:2]
    size_gray = main_gray.shape
    main_shrinked = cv2.resize(main_color, size_gray[::-1])
    
    sub_grays = []
    sub_shrinked = []
    for sub_color in sub_colors:
        shrinked = utility.shrink_img(sub_color)
        sub_shrinked.append(shrinked)
        sub_grays.append(cv2.cvtColor(shrinked, cv2.COLOR_BGR2GRAY))

    # state: 0 (完全に白飛び) ~ 1 (全く白飛びしていない)
    main_state = 1 - detect_overexposed(main_shrinked)
    main_state = cv2.erode(main_state, np.ones((50, 50)))  # 白飛び領域を十分に捉える
    main_state = cv2.blur(main_state, (50, 50))  # 境界をぼかす

    # tmp = cv2.cvtColor(main_gray, cv2.COLOR_GRAY2BGR)
    # tmp[:, :, 2] = main_state * 255
    # cv2.imshow(f"main_state", tmp)

    sub_states = []  
    for i in range(len(sub_grays)):
        state = np.ones(size_gray, dtype=np.float32)
        state = 1 - detect_overexposed(sub_shrinked[i])
        state = cv2.erode(state, np.ones((10, 10)))    
        state = cv2.blur(state, (50, 50))
        sub_states.append(state)
        # cv2.imshow(f"original_{i}", imgs[i])
    sub_states = np.array(sub_states, dtype=np.float32)

    sub_states_max = np.max(sub_states, axis=0)
    main_weight = np.clip((np.tanh((main_state - 0.93) * 15) + 1.0) * 0.5 + (np.tanh((main_state/sub_states_max - 1.0) * 1.0) + 1.0) * 0.3, 1e-6, 1)

    sub_weights_sum = np.sum(np.exp((sub_states - 0.95) * 5), axis=0)
    sub_weights = []
    for state in sub_states:
        sub_weight = (1 - main_weight) * np.exp((state - 0.95) * 5) / sub_weights_sum
        sub_weights.append(sub_weight)

    # 初めに小さい画像でブレンドして、明度情報を求めておく
    bld = main_shrinked * main_weight[:, :, np.newaxis]
    for img, weight in zip(sub_shrinked, sub_weights):
        bld = cv2.addWeighted(bld, 1.0, img * weight[:, :, np.newaxis], 1.0, 0)

    # tmp = main_shrinked.copy()
    # tmp[:, :, 2] = main_weight * 255
    # cv2.imshow(f"main_weight", tmp)
    # for i in range(len(sub_weights)):
    #     tmp = sub_shrinked[i].copy()
    #     tmp[:, :, 2] = sub_weights[i] * 255
    #     cv2.imshow(f"weight_{i}", tmp)

    # tmp = np.array(bld, dtype=np.uint8)
    # cv2.imshow("blended_dark", tmp)

    # HSV画像を用意(明度情報を使う)
    bld_hsv = cv2.cvtColor(bld, cv2.COLOR_BGR2HSV)
    sub_vs = []
    for img in sub_shrinked:
        sub_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sub_vs.append(sub_hsv[:, :, 2])

    # 暗い領域を明るくするため、明るくしてよい余裕を求める
    bright_margin = np.zeros(size_gray, dtype=np.float32)
    for img_v, state in zip(sub_vs, sub_states):
        bright_margin = np.maximum((img_v - bld_hsv[:, :, 2]) * (np.tanh((state - 0.99) * 15) + 1.0) * 0.3, bright_margin)

    # ぼかしをかけることで境目を目立たなくさせる
    bright_margin = cv2.blur(bright_margin, (20, 20))

    img_rgb = main_shrinked
    img_rgb[:, :, 2] = np.clip(bright_margin * 10, 0, 255)
    # cv2.imshow("bright_margin", img_rgb)

    main_weight = cv2.resize(main_weight, size_color[::-1])   # もとの画像サイズに合わせる
    blended = main_color * main_weight[:, :, np.newaxis]
    for img, weight in zip(sub_colors, sub_weights):
        weight = cv2.resize(weight, size_color[::-1])
        blended = cv2.addWeighted(blended, 1.0, img * weight[:, :, np.newaxis], 1.0, 0)

    bright_margin = cv2.resize(bright_margin, size_color[::-1])
    blended_hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
    blended_hsv[:, :, 2] = np.clip(blended_hsv[:, :, 2] + bright_margin, 0, 255)
    blended = cv2.cvtColor(blended_hsv, cv2.COLOR_HSV2BGR)

    return blended.astype(np.uint8)

def detect_overexposed(img):
    white = np.array([255, 255, 255])
    dist = np.max(white - img, axis=2)
    img_size = img.shape[:2]

    oe_area = np.zeros(img_size, dtype=np.float32)
    oe_area[dist <= 5] = 1.0
    if np.sum(oe_area) == 0:
        return oe_area
    first_dist_mean = np.mean(dist[oe_area == 1.0])
    allowd_dist_max = min(max(first_dist_mean, 1) * 7.5, 30)

    for t in range(10):
        oe_tmp = cv2.dilate(oe_area, np.ones((img_size[0]//5, img_size[1]//5)))
        mask = (oe_area == 0) & (oe_tmp == 1.0)  # 新しく白飛びと判定された領域
        if np.sum(mask) == 0: break
        dist_new = dist[mask]
        oe_new = oe_tmp[mask]
        indice = np.argsort(dist_new)
        dist_new_max = dist_new[indice[len(indice)//5]]     # 上位20%のみ採用する
        oe_new[dist_new > min(dist_new_max, allowd_dist_max)] = 0
        oe_area[mask] = oe_new
        # print(f"t = {t}")
        # print("new dist max", dist_new_max)
        # cv2.imshow(f"oe_area", oe_area)
        # cv2.waitKey()
        if dist_new_max > allowd_dist_max:
            break
    return oe_area
