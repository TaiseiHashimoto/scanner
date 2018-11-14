import cv2
import numpy as np
import utility

def align(main_gray, sub_colors):
    size_gray = main_gray.shape
    sub_grays = []
    sub_shrinkeds = []
    for sub_color in sub_colors:
        sub_shrinked = utility.shrink_img(sub_color)
        sub_shrinkeds.append(sub_shrinked)
        sub_grays.append(cv2.cvtColor(sub_shrinked, cv2.COLOR_BGR2GRAY))

    # AKAZEによるおおまかな位置合わせ
    akaze = cv2.AKAZE_create()
    akaze.setThreshold(0.0003)
    main_kps, main_dscs = akaze.detectAndCompute(main_gray, None)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)

    sub_aligneds = []    # 位置合わせ済みの画像
    for i in range(len(sub_colors)):
        kps, dscs = akaze.detectAndCompute(sub_grays[i], None)
        matches = bf.match(main_dscs, dscs)
        matches.sort(key=lambda m: m.distance)
        num_good_matches = min(int(len(matches) * 0.3), 100)
        if num_good_matches < 10: num_good_matches = len(matches)
        assert num_good_matches >= 4, f"Num of matches is too low. ({num_good_matches})"
        # print(f"Num of matches: {len(matches)}")
        # print(f"Num of good matches: {num_good_matches}")
        matches = matches[:num_good_matches]

        tmp = cv2.drawMatches(main_gray, main_kps, sub_grays[i], kps, matches, None)
        cv2.imwrite("matches.jpeg", tmp)

        main_pnts = []
        pnts = []
        for j in range(num_good_matches):
            main_pnts.append(main_kps[matches[j].queryIdx].pt)
            pnts.append(kps[matches[j].trainIdx].pt)
        main_pnts = np.array(main_pnts)
        pnts = np.array(pnts)

        M = cv2.findHomography(pnts, main_pnts, cv2.RANSAC)[0].astype(np.float32)
        
        # ECCによる精密な位置合わせ
        number_of_iterations = 10
        termination_eps = 1e-6
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        try:
            cc, M = cv2.findTransformECC(sub_grays[i], main_gray, M, cv2.MOTION_HOMOGRAPHY, criteria)
            # print(cc)
        except cv2.error:   # ECCが収束しなかった場合
            print("ECC stopped before convergence")
            continue

        aligned = cv2.warpPerspective(sub_shrinkeds[i], M, size_gray[::-1])

        if cc > 0.6:    # うまくマッチングできない画像は除外する
            sub_aligneds.append(aligned)

    return sub_aligneds

def detect_overexposed(img):
    white = np.array([255, 255, 255])
    dist = np.max(white - img, axis=2)
    img_size = img.shape[:2]

    oe_area = np.zeros(img_size, dtype=np.float32)
    oe_area[dist <= 10] = 1.0
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

    oe_area = cv2.dilate(oe_area, np.ones((50, 50)))  # 白飛び領域を十分に捉える
    oe_area = cv2.blur(oe_area, (50, 50))  # 境界をぼかす

    # tmp = utility.draw_oearea(oe_area)
    # cv2.imwrite("oe_area.jpeg", tmp)
    # cv2.imshow("oe_area", tmp)
    # cv2.waitKey()
    # exit(0)

    return oe_area
