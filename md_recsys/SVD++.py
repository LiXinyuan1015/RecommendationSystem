from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
import numpy as np
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
data = Dataset.load_from_file('data/train.csv', reader=reader)
train_set = data.build_full_trainset()
model = SVDpp()
kf = KFold(n_splits=3) # 3折交叉验证
for train_data, dev_data in kf.split(data):
    model.fit(train_data)
    predictions = model.test(dev_data)
    rmse = accuracy.rmse(predictions, verbose=True) #计算RMSE
save_results = list()
dtype = [("userId", np.int32), ("movieId", np.int32)]
test_data = pd.read_csv("data/test.csv", usecols=range(1,3), dtype=dict(dtype))
test_data = pd.DataFrame(test_data)
for uid, iid in test_data.itertuples(index=False):
    pred = model.predict(uid, iid, r_ui=4, verbose=True)
    save_results.append(pred)