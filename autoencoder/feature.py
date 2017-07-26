import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.models import model_from_json
import keras.backend as K
from LoadData import load_test, load_train, shuffle_data


col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 'Voltage 2', 'Voltage 3', 'Accelerometer 1', 'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
train, label= load_train(col, '/media/meng/9079-7B0D/clean_data/clean/train/', 6)
test, test_y = load_test(col, "/media/meng/9079-7B0D/clean_data/clean/test/")
train, label = shuffle_data(train, label)


model = model_from_json(open('/home/meng/PyProject/Motor-life-prediction/autoencoder/model.json').read())
model.load_weights('/home/meng/PyProject/Motor-life-prediction/autoencoder/model.h5')

get_feature = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])
test_feature = get_feature([test, 0])
train_feature = get_feature([train, 0])

print np.shape(test_feature)

cols = []
for i in range(256):
    cols.append("feature_"+str(i))

test_feature = pd.DataFrame(columns=cols, data=test_feature[0])
test_feature['label'] = test_y
test_feature.to_csv('/home/meng/PyProject/Motor-life-prediction/feature_data/autoencide_test.csv', index=False)

train_feature = pd.DataFrame(columns=cols, data=train_feature[0])
train_feature['label'] = label
train_feature.to_csv("/home/meng/PyProject/Motor-life-prediction/feature_data/autoencode_train.csv", index=False)
