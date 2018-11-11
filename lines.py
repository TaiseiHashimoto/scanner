import numpy as np
from itertools import combinations
import utility

class Lines:
    def __init__(self, points):
        self.points = np.array(points, dtype=np.float32)
        self.theta = np.arctan((points[:, 1] - points[:, 3]) / (points[:, 0] - points[:, 2] + 1e-6))
        mx = (self.points[:, 0] + self.points[:, 2]) * 0.5
        my = (self.points[:, 1] + self.points[:, 3]) * 0.5
        self.mupnts = np.hstack((mx[:, np.newaxis], my[:, np.newaxis]))
        self.num = len(points)

    def remove(arr):
        self.points = self.points[~arr]
        self.theta = self.theta[~arr]
        self.mupnts = self.mupnts[~arr]

    def remove_central(self, img_size, ratio=0.25):
        mx = self.mupnts[:, 0]
        my = self.mupnts[:, 1]
        cond = (mx < img_size[1] * ratio) | (mx > img_size[1] * (1-ratio)) | \
                (my < img_size[0] * ratio) | (my > img_size[0] * (1-ratio))
        self.points = self.points[cond]
        self.theta = self.theta[cond]
        self.mupnts = self.mupnts[cond]
        self.num = np.sum(cond)

    def equal(self):
        indice = np.array(list(combinations(range(self.num), 2)))
        equal = np.ones((len(indice),), dtype=np.bool)

        l1 = self.points[indice[:, 0]]
        l2 = self.points[indice[:, 1]]
        th1 = self.theta[indice[:, 0]]
        th2 = self.theta[indice[:, 1]]

        th_delta = utility.angle_normalize(th1 - th2)
        th_avg = utility.angle_normalize(th2 + th_delta / 2)
        th_delta = np.abs(th_delta)

        line_equal_deg = 2
        equal[th_delta * 180 > line_equal_deg * np.pi] = False

        # point_connect_distance = 50
        point_connect_distance = 52.5
        dist_point = np.min([ \
            np.power(l1[:, 0] - l2[:, 0], 2) + np.power(l1[:, 1] - l2[:, 1], 2), \
            np.power(l1[:, 0] - l2[:, 2], 2) + np.power(l1[:, 1] - l2[:, 3], 2), \
            np.power(l1[:, 2] - l2[:, 0], 2) + np.power(l1[:, 3] - l2[:, 1], 2), \
            np.power(l1[:, 2] - l2[:, 2], 2) + np.power(l1[:, 3] - l2[:, 3], 2) \
            ], axis=0)
        dist_point = np.sqrt(dist_point)
        equal[dist_point > point_connect_distance] = False

        # line_equal_distance = 2.5
        line_equal_distance = 2.625
        mx1 = (l1[:, 0] + l1[:, 2]) / 2
        my1 = (l1[:, 1] + l1[:, 3]) / 2
        mx2 = (l2[:, 0] + l2[:, 2]) / 2
        my2 = (l2[:, 1] + l2[:, 3]) / 2
        dist_ver = np.abs(np.sin(th_avg) * (mx1 - mx2) - np.cos(th_avg) * (my1 - my2))
        equal[dist_ver > line_equal_distance] = False

        return equal