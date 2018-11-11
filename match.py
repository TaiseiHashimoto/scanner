import cv2
import numpy as np


def homography(main_img, sub_imgs):
    main_gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    sub_grays = [cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) for sub in sub_imgs]
    img_size = main_gray.shape

    akaze = cv2.AKAZE_create()
    main_kps, main_dscs = akaze.detectAndCompute(main_gray, None)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)

    warped_imgs = []
    matched_pnts = []    # blendにおいて使用する

    for idx, [img, gray] in enumerate(zip(sub_imgs, sub_grays)):
        kps, dscs = akaze.detectAndCompute(gray, None)

        matches = bf.match(main_dscs, dscs)
        matches.sort(key=lambda m: m.distance)

        num_good_matches = min(int(len(matches) * 0.1), 100)
        print(f"Num of matches: {len(matches)}")
        print(f"Num of good matches: {num_good_matches}")
        matches = matches[:num_good_matches]

        # img_matching = cv2.drawMatches(main_img, main_kps, img, kps, matches, None)
        # cv2.imshow(f"matches_{idx}", img_matching)

        main_pnts = []
        pnts = []
        for i in range(num_good_matches):
            main_pnts.append(main_kps[matches[i].queryIdx].pt)
            pnts.append(kps[matches[i].trainIdx].pt)
        main_pnts = np.array(main_pnts)
        pnts = np.array(pnts)
        matched_pnts.append(main_pnts)

        trans_mat = cv2.findHomography(pnts, main_pnts, cv2.RANSAC)[0]
        warped_img = cv2.warpPerspective(img, trans_mat, gray.shape[::-1])
        warped_imgs.append(warped_img)
        idx += 1

    return warped_imgs, matched_pnts


def blend(main_img, sub_imgs, matched_pnts):
    img_shape = main_img.shape

    # state: 0 (完全に白飛び) ~ 1 (全く白飛びしていない)
    main_state = 1 - detect_overexposed(main_img)
    main_state = cv2.erode(main_state, np.ones((50, 50)))  # 白飛び領域を十分に捉える
    main_state = cv2.blur(main_state, (50, 50))  # 境界をぼかす

    img_rgb = main_img.copy()
    img_rgb[:, :, 2] = main_state * 255
    cv2.imshow(f"main_state", img_rgb)

    sub_states = []  
    for img in sub_imgs:
        state = np.ones(img_shape[:2], dtype=np.float32)
        state = 1 - detect_overexposed(img)
        state = cv2.erode(state, np.ones((10, 10)))    
        state = cv2.blur(state, (50, 50))
        sub_states.append(state)
        # cv2.imshow(f"original_{i}", imgs[i])
    sub_states = np.array(sub_states, dtype=np.float32)

    sub_states_max = np.max(sub_states, axis=0)
    main_weight = np.clip((np.tanh((main_state - 0.93) * 15) + 1.0) * 0.5 + (np.tanh((main_state/sub_states_max - 1.0) * 1.0) + 1.0) * 0.3, 1e-6, 1)

    sub_weights = []
    sub_weights_sum = np.sum(np.exp((sub_states - 0.95) * 5), axis=0)
    for state in sub_states:
        sub_weights.append((1 - main_weight) * np.exp((state - 0.95) * 5) / sub_weights_sum)

    blended = main_img * main_weight[:, :, np.newaxis]
    for img, weight in zip(sub_imgs, sub_weights):
        blended = cv2.addWeighted(blended, 1.0, img * weight[:, :, np.newaxis], 1.0, 0)

    img_rgb = main_img.copy()
    img_rgb[:, :, 2] = main_weight * 255
    cv2.imshow(f"main_weight", img_rgb)
    for idx, [img, weight] in enumerate(zip(sub_imgs, sub_weights)):
        img_rgb = img.copy()
        img_rgb[:, :, 2] = weight * 255
        cv2.imshow(f"weight_{idx}", img_rgb)

    comp = np.array(blended, dtype=np.uint8)
    cv2.imshow("blended_dark", comp)

    # 暗い領域を明るくする
    brighten = np.zeros(img_shape[:2], dtype=np.float32)
    comp_hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
    comp_v = comp_hsv[:, :, 2]
    for img, state in zip(sub_imgs, sub_states):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_v = img_hsv[:, :, 2]
        brighten = np.maximum((img_v - comp_v) * (np.tanh((state - 0.99) * 15) + 1.0) * 0.7, brighten)

    # ぼかしをかけることでノイズを低減
    brighten = cv2.blur(brighten, (20, 20))
    comp_hsv[:, :, 2] = np.maximum(comp_v, np.clip(comp_v + brighten, 0, 255))

    blended = cv2.cvtColor(comp_hsv, cv2.COLOR_HSV2BGR)
    blended = np.array(blended, dtype=np.uint8)

    img_rgb = main_img
    img_rgb[:, :, 2] = np.clip(brighten * 10, 0, 255)
    cv2.imshow("brighten", img_rgb)

    return blended


def detect_overexposed(img):
    white = np.array([255, 255, 255])
    dist = np.max(white - img, axis=2)

    oe_area = np.zeros(img.shape[:2], dtype=np.float32)
    oe_area[dist <= 5] = 1.0
    if np.sum(oe_area) == 0:
        return oe_area
    first_dist_mean = np.mean(dist[oe_area == 1.0])
    allowd_dist_max = min(max(first_dist_mean, 1) * 7.5, 30)

    for t in range(10):
        oe_tmp = cv2.dilate(oe_area, np.ones((img.shape[0]//5, img.shape[1]//5)))
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
