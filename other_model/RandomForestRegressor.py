from Load_Data import load_data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def get_model():
    RFR =  RandomForestRegressor(n_estimators=10, max_depth=10, criterion="mae", n_jobs=1)
    return RFR

RFR = get_model()

train_path = '/home/meng/PyProject/Motor-life-prediction/feature_data/train.csv'
test_path = '/home/meng/PyProject/Motor-life-prediction/feature_data/test.csv'
train_x, train_y = load_data(train_path)
test_x, test_y = load_data(test_path)
print np.shape(train_x)
print np.shape(train_y)

RFR.fit(train_x, train_y)
ret = RFR.predict(test_x)

predict_ret = pd.DataFrame({'predicit': ret})
predict_ret.to_csv("/home/meng/PyProject/Motor-life-prediction/feature_data/predict.csv", index=False)
