import pickle
import pandas as pd
import numpy as np
import sys

with open("ml_model.pickle", "rb") as f:
	model = pickle.load(f)

data = pd.read_csv(sys.argv[1]).values[:, 2:];
scores = model.predict_proba(data)
ids = list(range(len(data)))
result = np.vstack(ids, scores)

np.savetxt("score.csv", result, delimiter=',')
