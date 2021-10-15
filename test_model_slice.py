
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

sample, m = 1000, 10
# rx, ry = np.random.rand(sample, m), np.random.rand(sample)
# np.savez('test_xgb.npz', rx=rx, ry=ry)

d = np.load('test_xgb.npz')
rx, ry = d['rx'][:], d['ry'][:]

def get_slice_model(m):
    assert isinstance(m, xgb.Booster)
    if hasattr(m, 'best_ntree_limit') and m.best_ntree_limit > 0:
        n = m.best_ntree_limit + 1
        print("best_ntree_limit", n, m.best_iteration)
        return m[0:n]
    return m


def train_model(rx, ry):
    rx, rx_v, ry, ry_v = train_test_split(rx, ry, test_size=0.2, shuffle=False)

    dtrain = xgb.DMatrix(rx, label=ry)
    dval = xgb.DMatrix(rx_v, label=ry_v)

    m = xgb.train({'objective': 'reg:squarederror'}, dtrain, 200, evals=[(dval, "validation")],
                  early_stopping_rounds=20, verbose_eval=False)
    print(m.attributes())
    m.predict(dtrain)
    m = get_slice_model(m)
    # m.predict(dtrain)
    print('inner:', m.num_boosted_rounds(), m.num_features())
    return m


def run_test():
    m = train_model(rx, ry)
    print('outer:', m.num_boosted_rounds(), m.num_features())
    m.inplace_predict(rx)


if __name__ == '__main__':
    run_test()
