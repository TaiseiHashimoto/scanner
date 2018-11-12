import numpy as np
import utility

class Segments:
    def __init__(self, lines, labels, labels_num, img_size):
        self.theta = []
        self.pnt1 = []
        self.pnt2 = []
        self.mupnts = []
        self.expnt1 = []
        self.expnt2 = []
        self.exmupnts = []
        self.is_hor = []
        self.is_ver = []
        self.num = labels_num
        for l in range(labels_num):
            th = utility.angle_mean(lines.theta[labels == l])
            self.theta.append(th)
            mupnt = np.mean(lines.mupnts[labels == l], axis=0)
            self.mupnts.append(mupnt)

            line_lens = []
            direction = np.array((np.cos(th), np.sin(th)))
            line_lens.extend(np.dot(lines.points[labels == l][:, :2] - mupnt, direction))
            line_lens.extend(np.dot(lines.points[labels == l][:, 2:] - mupnt, direction))
            len_min = np.min(line_lens)
            len_max = np.max(line_lens)

            pnt1 = mupnt + direction * len_min
            pnt2 = mupnt + direction * len_max
            ext_len = max(img_size)    * 2  # 画面からはみ出る十分な長さ
            # 縦横の順序に注意
            expnt1 = np.clip(mupnt - direction * ext_len, (0, 0), img_size[::-1])
            expnt2 = np.clip(mupnt + direction * ext_len, (0, 0), img_size[::-1])
            exmupnt = (expnt1 + expnt2) * 0.5

            if -np.pi/4 < th < np.pi/4:
                hor, ver = True, False
            else:
                hor, ver = False, True
                if th < 0:
                    pnt1, pnt2 = pnt2, pnt1
                    expnt1, expnt2 = expnt2, expnt1

            self.pnt1.append(pnt1)
            self.pnt2.append(pnt2)
            self.expnt1.append(expnt1)
            self.expnt2.append(expnt2)
            self.exmupnts.append(exmupnt)
            self.is_hor.append(hor)
            self.is_ver.append(ver)

        self.theta = np.array(self.theta, dtype=np.float32)
        self.mupnts = np.array(self.mupnts, dtype=np.float32)
        self.exmupnts = np.array(self.exmupnts, dtype=np.float32)
        self.pnt1 = np.array(self.pnt1, dtype=np.float32)
        self.pnt2 = np.array(self.pnt2, dtype=np.float32)
        self.expnt1 = np.array(self.expnt1, dtype=np.float32)
        self.expnt2 = np.array(self.expnt2, dtype=np.float32)
        self.is_hor = np.array(self.is_hor, dtype=np.bool)
        self.is_ver = np.array(self.is_ver, dtype=np.bool)

    def remove_central(self, img_size, ratio=0.2):
        mx = self.exmupnts[:, 0]
        my = self.exmupnts[:, 1]
        cond = (mx < img_size[1] * ratio) | (mx > img_size[1] * (1-ratio)) | \
                (my < img_size[0] * ratio) | (my > img_size[0] * (1-ratio))
        self.theta = self.theta[cond]
        self.pnt1 = self.pnt1[cond]
        self.pnt2 = self.pnt2[cond]
        self.mupnts = self.mupnts[cond]
        self.expnt1 = self.expnt1[cond]
        self.expnt2 = self.expnt2[cond]
        self.exmupnts = self.exmupnts[cond]
        self.is_hor =  self.is_hor[cond]
        self.is_ver =  self.is_ver[cond]
        self.num = np.sum(cond)