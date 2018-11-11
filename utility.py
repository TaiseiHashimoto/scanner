import cv2
import numpy as np

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
        img = cv2.putText(img, str(i), (mupnt[0], mupnt[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, c)
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
    if size[0] > size[1] and size[0] > 600:
        img = cv2.resize(img, (600*size[1]//size[0], 600))
    elif size[0] < size[1] and size[1] > 600:
        img = cv2.resize(img, (600, 600*size[0]//size[1]))
    return img