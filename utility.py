import cv2
import numpy as np
import pandas as pd
import itertools

def angle_normalize(theta):
    theta = np.array(theta, dtype=np.float32)
    theta[theta < -np.pi / 2] += np.pi
    theta[theta > np.pi / 2] -= np.pi
    return theta

def angle_mean(theta):
    theta = np.array(theta, dtype=np.float32)
    # -pi/2, pi/2付近に分かれている場合
    if np.any(theta < -np.pi / 4) and np.any(theta > np.pi / 4):
        theta[theta < -np.pi/4] += np.pi
    
    th_mean = np.mean(theta)
    if th_mean > np.pi / 2:
        return th_mean - np.pi
    elif th_mean < -np.pi / 2:
        return th_mean + np.pi
    return th_mean

def draw_lines(img, lines, labels, labels_num):
    colors = np.random.randint(255, size=(labels_num, 3))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pnt, label in zip(lines.points, labels):
        c = (int(colors[label][0]), int(colors[label][1]), int(colors[label][2]))
        img = cv2.line(img, (pnt[0], pnt[1]), (pnt[2], pnt[3]), c, 3)
    return img

def draw_segments(img, segments):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = np.random.randint(255, size=(segments.num, 3))
    for i in range(segments.num):
        c = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        pnt1 = segments.pnt1[i]
        pnt2 = segments.pnt2[i]
        img = cv2.line(img, (pnt1[0], pnt1[1]), (pnt2[0], pnt2[1]), c, 2)
    for i in range(segments.num):
        c = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        mupnt = segments.mupnts[i]
        pnt1 = segments.pnt1[i]
        pnt2 = segments.pnt2[i]
        img = cv2.putText(img, str(i), (mupnt[0], int(mupnt[1] - 3)), cv2.FONT_HERSHEY_PLAIN, 1.0, c)
        img = cv2.circle(img, (pnt1[0], pnt1[1]), 3, c, -1)
        img = cv2.circle(img, (pnt2[0], pnt2[1]), 3, c, -1)
    return img

def draw_intersections(img, intersections, indice=None):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if indice is None:
        indice = range(intersections.num)
    for idx in indice:
        pnt = intersections.cross_pnt[idx]
        img = cv2.circle(img, (pnt[0], pnt[1]), 1, (255, 0, 0), -1)
        img = cv2.putText(img, str(idx), (pnt[0], pnt[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))
    return img

def draw_detected(img, intersections, idx_lt, idx_rt, idx_lb, idx_rb):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pnt_lt = intersections.cross_pnt[idx_lt]
    pnt_rt = intersections.cross_pnt[idx_rt]
    pnt_lb = intersections.cross_pnt[idx_lb]
    pnt_rb = intersections.cross_pnt[idx_rb]
    img = cv2.circle(img, (pnt_lt[0], pnt_lt[1]), 3, (0, 0, 255), -1)
    img = cv2.circle(img, (pnt_rt[0], pnt_rt[1]), 3, (0, 0, 255), -1)
    img = cv2.circle(img, (pnt_lb[0], pnt_lb[1]), 3, (0, 0, 255), -1)
    img = cv2.circle(img, (pnt_rb[0], pnt_rb[1]), 3, (0, 0, 255), -1)
    return img

def shrink_img(img):
    size = img.shape
    length = 600
    if size[0] > size[1] and size[0] > length:
        img = cv2.resize(img, (int(length*size[1]/size[0]), length), cv2.INTER_AREA)
    elif size[0] < size[1] and size[1] > length:
        img = cv2.resize(img, (length, int(length*size[0]/size[1])), cv2.INTER_AREA)
    return img


def get_best_set(intersections, all_scores, vertex_lt, vertex_rt, vertex_lb, vertex_rb):
    # indice = np.array(list(itertools.product(vertex_lt, vertex_rt, vertex_lb, vertex_rb)))
    # v1 = intersections.cross_pnt[indice[:, 1]] - intersections.cross_pnt[indice[:, 0]]
    # v2 = intersections.cross_pnt[indice[:, 3]] - intersections.cross_pnt[indice[:, 1]]
    # v3 = intersections.cross_pnt[indice[:, 2]] - intersections.cross_pnt[indice[:, 3]]
    # v4 = intersections.cross_pnt[indice[:, 0]] - intersections.cross_pnt[indice[:, 2]]

    # v1_len = np.linalg.norm(v1)
    # v2_len = np.linalg.norm(v2)
    # v3_len = np.linalg.norm(v3)
    # v4_len = np.linalg.norm(v4)

    # deg1 = np.arccos(np.sum(v1 * v2, axis=1) / (v1_len * v2_len)) - np.pi / 2
    # deg2 = np.arccos(np.sum(v2 * v3, axis=1) / (v2_len * v3_len)) - np.pi / 2
    # deg3 = np.arccos(np.sum(v3 * v4, axis=1) / (v3_len * v4_len)) - np.pi / 2
    # deg4 = np.arccos(np.sum(v4 * v1, axis=1) / (v4_len * v1_len)) - np.pi / 2

    # scores = np.sum(all_scores[indice], axis=1)
    # degree_loss = np.abs(deg1) + np.abs(deg2) + np.abs(deg3) + np.abs(deg4)
    # weight = 200
    # print(f"org_score: \n{scores}")
    # print(f"degs: \n{degree_loss * weight}")
    # scores -= degree_loss * weight

    # best_idx = np.argmax(scores)

    # return indice[best_idx]

    return vertex_lt[0], vertex_rt[0], vertex_lb[0], vertex_rb[0]