import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_score(df):
    with open("ml_model.pickle", "rb") as f:
        model = pickle.load(f)
    data = df.iloc[:, 2:].astype(np.float64)
    scores = model.predict_proba(data)[:, 1]  # ラベル1(positive)の確率
    return scores

def prepare_data(intersections, img_size):
    height, width = img_size
    data = pd.DataFrame()
    data['id'] = list(range(intersections.num))
    data['posnum'] = intersections.posnum
    data['seglen_hor'] = intersections.seglen_hor / width
    data['seglen_ver'] = intersections.seglen_ver / height
    data['segdist_hor'] = intersections.segdist_hor / width
    data['segdist_ver'] = intersections.segdist_ver / height
    data['segdeg_hor'] = np.abs(intersections.segdeg_hor)
    data['segdeg_ver'] = np.abs(intersections.segdeg_ver)
    data['cross_deg'] = intersections.cross_deg
    data['dist_hor'] = intersections.dist_hor
    data['dist_ver'] = intersections.dist_ver
    return data