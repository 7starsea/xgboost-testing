
import numpy as np
from XgbShannon import XgbShannonPredictor
import xgboost as xgb


rx = np.random.rand(100, 127 + 7 + 1)
rx = rx.astype(np.float32, order='C')
rx2 = np.array(rx, copy=True)

fname = 'xgb.model.bin'  # # trained from v1.2.1 python

m2 = xgb.Booster({'nthread': '4'})  # init model
m2.load_model(fname)  # load data

dtest = xgb.DMatrix(rx, missing=0.0)
y1 = m2.predict(dtest)

m1 = XgbShannonPredictor(fname)

y2 = m1.predict(rx2)

z = np.abs(y1 - y2)
print('difference:', np.percentile(z, [10, 50, 90, 99]), np.sum(z), np.max(z))
