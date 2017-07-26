import xgboost as xgb
from Load_Data import load_data
import numpy as np
import pandas as pd

train_path = '/home/meng/PyProject/Motor-life-prediction/feature_data/autoencode_train.csv'
test_path = '/home/meng/PyProject/Motor-life-prediction/feature_data/autoencode_test.csv'
train_x, train_y = load_data(train_path)
test_x, test_y = load_data(test_path)

pd.DataFrame({'label': test_y}).to_csv('/home/meng/PyProject/Motor-life-prediction/feature_data/new_label.csv', index=False)

#train = xgb.DMatrix(train_x, train_y)
#test = xgb.DMatrix(test_x)
# 0.5
reg = xgb.XGBRegressor(objective="reg:linear", 
booster='gbtree', subsample=0.5, colsample_bylevel=0.1, max_depth=8, n_estimators=200)


reg.fit(train_x, train_y, eval_metric='mae', verbose = True, eval_set = [(test_x,test_y)], early_stopping_rounds=5)
ret = reg.predict(test_x, ntree_limit=reg.best_iteration)
print np.shape(ret)
predict_ret = pd.DataFrame({'predicit': ret})
predict_ret.to_csv("/home/meng/PyProject/Motor-life-prediction/feature_data/xgboost1.csv", index=False)
