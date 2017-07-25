from Load_Data import load_data
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

def get_model():
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=0, loss='lad')
    return model
RFR = get_model()

train_path = '/home/meng/PyProject/Motor-life-prediction/feature_data/train.csv'
test_path = '/home/meng/PyProject/Motor-life-prediction/feature_data/test.csv'
train_x, train_y = load_data(train_path)
test_x, test_y = load_data(test_path)
print np.shape(train_x)
print np.shape(train_y)

RFR.fit(train_x, train_y)
ret = RFR.predict(test_x)
joblib.dump(RFR, '/home/meng/PyProject/Motor-life-prediction/RandomForest/model.m')

predict_ret = pd.DataFrame({'predicit': ret})
predict_ret.to_csv("/home/meng/PyProject/Motor-life-prediction/feature_data/predict.csv", index=False)
