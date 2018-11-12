import numpy as np
import utility

class Intersections:
    def __init__(self, segments, img_size, ratio=0.2):
        f = open("report_intersection.txt", "w")

        img_height, img_width = img_size
        horizontal = np.where(segments.is_hor)[0]
        vertical = np.where(segments.is_ver)[0]
        indice = np.array(np.meshgrid(horizontal, vertical)).T.reshape((-1, 2))
        indice_len = indice.shape[0]
        valid = np.ones((indice_len,), dtype=np.bool)

        th_hor = segments.theta[indice[:, 0]]
        th_ver = segments.theta[indice[:, 1]]
        hor_pnt1 = segments.pnt1[indice[:, 0]]
        hor_pnt2 = segments.pnt2[indice[:, 0]]
        ver_pnt1 = segments.pnt1[indice[:, 1]]
        ver_pnt2 = segments.pnt2[indice[:, 1]]

        line_cross_degree = 70
        cross_deg = np.empty((indice_len,), dtype=np.float32)
        segdeg_hor = np.empty((indice_len,), dtype=np.float32)
        segdeg_ver = np.empty((indice_len,), dtype=np.float32)
        cross_deg = np.abs(utility.angle_normalize(th_hor - th_ver))
        segdeg_hor = th_hor
        segdeg_ver = th_ver

        old_valid = valid.copy()
        valid &= cross_deg * 180 > line_cross_degree * np.pi
        for idx in np.where(old_valid & ~valid)[0]:
            f.write(f"{indice[idx, 0]}, {indice[idx, 1]}  rejected (cross_deg[{cross_deg[idx]}] is small)\n")

        hor_pnt1 = hor_pnt1[valid]
        ver_pnt1 = ver_pnt1[valid]

        x = ver_pnt1[valid] - hor_pnt1[valid]
        d1 = hor_pnt2[valid] - hor_pnt1[valid]
        d2 = ver_pnt2[valid] - ver_pnt1[valid]
        t = np.cross(x, d2) / np.cross(d1, d2)
        cross_pnt = np.empty((indice_len, 2), dtype=np.float32)
        cross_pnt[valid] = hor_pnt1[valid] + (d1.T * t).T

        old_valid = valid.copy()
        valid &= (cross_pnt[:, 0] >= 0) & \
                (cross_pnt[:, 0] <= img_width) & \
                (cross_pnt[:, 1] >= 0) & \
                (cross_pnt[:, 1] <= img_height)
        for idx in np.where(old_valid & ~valid)[0]:
            f.write(f"{indice[idx, 0]}, {indice[idx, 1]}  rejected (cross point{cross_pnt[idx]} is in center or out of image)\n")

        left = cross_pnt[:, 0] < np.minimum(hor_pnt2[:, 0], img_width * ratio)
        right = cross_pnt[:, 0] > np.maximum(hor_pnt1[:, 0], img_width * (1-ratio))
        is_left = np.zeros((indice_len,), dtype=np.bool)
        is_right = np.zeros((indice_len,), dtype=np.bool)
        is_left[valid & left] = True
        is_right[valid & right] = True
        segdist_hor = np.zeros((indice_len,), dtype=np.float32)
        segdist_hor[valid & left] = np.linalg.norm(hor_pnt1[valid & left] - cross_pnt[valid & left], axis=1)
        segdist_hor[valid & right] = np.linalg.norm(hor_pnt2[valid & right] - cross_pnt[valid & right], axis=1)

        old_valid = valid.copy()
        valid &= left | right
        for idx in np.where(old_valid & ~valid)[0]:
            f.write(f"{indice[idx, 0]}, {indice[idx, 1]}  rejected (cross point{cross_pnt[idx]} is not either left or right)\n")
        old_valid = valid.copy()
        valid &= segdist_hor < img_width * 0.5
        for idx in np.where(old_valid & ~valid)[0]:
            f.write(f"{indice[idx, 0]}, {indice[idx, 1]} is_left[{is_left[idx]}] rejected (segdist_hor[{segdist_hor[idx]}] is too big)\n")

        top = cross_pnt[:, 1] < np.minimum(ver_pnt2[:, 1], img_height * ratio)
        bottom = cross_pnt[:, 1] > np.maximum(ver_pnt1[:, 1], img_height * (1-ratio))
        is_top = np.zeros((indice_len,), dtype=np.bool)
        is_bottom = np.zeros((indice_len,), dtype=np.bool)
        is_top[valid & top] = True
        is_bottom[valid & bottom] = True
        segdist_ver = np.zeros((indice_len,), dtype=np.float32)
        segdist_ver[valid & top] = np.linalg.norm(ver_pnt1[valid & top] - cross_pnt[valid & top], axis=1)
        segdist_ver[valid & bottom] = np.linalg.norm(ver_pnt2[valid & bottom] - cross_pnt[valid & bottom], axis=1)

        old_valid = valid.copy()
        valid &= top | bottom
        for idx in np.where(old_valid & ~valid)[0]:
            f.write(f"{indice[idx, 0]}, {indice[idx, 1]}  rejected (cross point{cross_pnt[idx]} is not either top or bottom)\n")
        old_valid = valid.copy()
        valid &= segdist_ver < img_height * 0.5
        for idx in np.where(old_valid & ~valid)[0]:
            f.write(f"{indice[idx, 0]}, {indice[idx, 1]}  is_top[{is_top[idx]}] rejected (segdist_ver[{segdist_ver[idx]}] is too big)\n")

        seglen_hor = np.empty((indice_len,), dtype=np.float32)
        seglen_ver = np.empty((indice_len,), dtype=np.float32)
        seglen_hor[valid] = np.linalg.norm(hor_pnt1[valid] - hor_pnt2[valid], axis=1)
        seglen_ver[valid] = np.linalg.norm(ver_pnt1[valid] - ver_pnt2[valid], axis=1)

        dist_hor = np.empty((indice_len,), dtype=np.float32)
        dist_ver = np.empty((indice_len,), dtype=np.float32)
        dist_hor[valid & is_left] = cross_pnt[valid & is_left, 0] / img_width
        dist_hor[valid & is_right] = 1 - cross_pnt[valid & is_right, 0] / img_width
        dist_ver[valid & is_top] = cross_pnt[valid & is_top, 1] / img_height
        dist_ver[valid & is_bottom] = 1 - cross_pnt[valid & is_bottom, 1] / img_height


        posnum = np.empty((indice_len,), dtype=np.uint8)
        posnum[valid & is_left & is_top] = 0
        posnum[valid & is_right & is_top] = 1
        posnum[valid & is_left & is_bottom] = 2
        posnum[valid & is_right & is_bottom] = 3

        for i, idx in enumerate(np.where(valid)[0]):
            f.write(f"{i}: {indice[idx, 0]}, {indice[idx, 1]}  posnum[{posnum[idx]}] position{cross_pnt[idx]}\n")

        self.num = np.sum(valid)
        self.posnum = posnum[valid]
        self.cross_deg = cross_deg[valid]
        self.segdeg_hor = segdeg_hor[valid]
        self.segdeg_ver = segdeg_ver[valid]
        self.cross_pnt = cross_pnt[valid]
        self.segdist_hor = segdist_hor[valid]
        self.segdist_ver = segdist_ver[valid]
        self.seglen_hor = seglen_hor[valid]
        self.seglen_ver = seglen_ver[valid]
        self.is_left = is_left[valid]
        self.is_right = is_right[valid]
        self.is_top = is_top[valid]
        self.is_bottom = is_bottom[valid]
        self.dist_hor = dist_hor[valid]
        self.dist_ver = dist_ver[valid]

        f.close()

def posnum(idx, is_left, is_right, is_top, is_bottom):
    if is_left[idx] and is_top[idx]:
        return 0
    if is_right[idx] and is_top[idx]:
        return 1
    if is_left[idx] and is_bottom[idx]:
        return 2
    if is_right[idx] and is_bottom[idx]:
        return 3