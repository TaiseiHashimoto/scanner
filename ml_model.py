import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def get_score(data):
	with open("ml_model.pickle", "rb") as f:
	    model = pickle.load(f)

	stdsc = StandardScaler()
	data = stdsc.fit_transform(data)
	# ラベル1(positive)の確率
	scores = model.predict_proba(data)[:, 1]
	return scores

def prepare_data(intersections, img_size):
	height, width = img_size
	n_features = 9
	data = np.empty((intersections.num, n_features), dtype=np.float32)
	data[:, 0] = intersections.seglen_hor / width
	data[:, 1] = intersections.seglen_ver / height
	data[:, 2] = intersections.segdist_hor / width
	data[:, 3] = intersections.segdist_ver / height
	data[:, 4] = intersections.cross_deg
	data[:, 5] = np.abs(intersections.segdeg_hor)
	data[:, 6] = np.abs(intersections.segdeg_ver)
	data[:, 7] = intersections.dist_hor
	data[:, 8] = intersections.dist_ver
	
	return data