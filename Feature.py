import os
import pandas as pd
import numpy
from sklearn import preprocessing
from keras.models import model_from_json
import keras.backend as K
from Load_Data import load_train, load_test, shuffle_data

'''
def load_data(col):
    path = "/media/meng/9079-7B0D/clean_m1/"
    filelist = os.listdir(path)
    ret = []
    label = []
    dicit = set()
    for filename in filelist:
        fullpath = path + filename
        data = pandas.read_csv(fullpath)
        data = data.loc[:, col].values
        rows = numpy.shape(data)[0]
        dicit.add(filename[0:15])
        lab = len(dicit) - 1
        for i in range(0, rows, 200):
            tmp = []
            for j in range(i, i+200):
                tmp.append(data[j, :])
            ret.append(tmp)
            label.append(lab)
    print len(dicit)
    ret = numpy.asarray(ret, dtype="float32")
    #print numpy.shape(ret)
    ret = numpy.reshape(ret, (len(filelist) * 100, -1))
    #print numpy.shape(ret)
    min_max_scaler = preprocessing.MinMaxScaler()
    ret = min_max_scaler.fit_transform(ret)
    #print data[0:3, 0:20]
    ret = numpy.reshape(ret, (len(filelist) * 100, 200, -1))
    #print numpy.shape(ret)
    label = numpy.array(label)
    #print numpy.shape(label)
    return ret, label
'''
col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 'Voltage 2', 'Voltage 3', 'Accelerometer 1', 'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
train, label= load_train(col, '/media/meng/9079-7B0D/clean_data/train/', 6)
test, test_y = load_test(col, "/media/meng/9079-7B0D/clean_data/test/")

train, label = shuffle_data(train, label)
test, test_y = shuffle_data(test, test_y)

#train, label = shuffle_data(train, label)

train = train.reshape(train.shape[0], 200, 13, 1)
test = test.reshape(test.shape[0], 200, 13, 1)

model = model_from_json(open('/home/meng/PyProject/Motor-life-prediction/cnn_line_model/std_model_5.json').read())
model.load_weights('/home/meng/PyProject/Motor-life-prediction/cnn_line_model/std_model_5.h5')
get_feature = K.function([model.layers[0].input, K.learning_phase()], [model.layers[9].output])

test_feature = get_feature([test, 0])
train_feature = get_feature([train, 0])
print numpy.shape(test_feature[0])

cols = []
for i in range(280):
    cols.append("feature_"+str(i))

test_feature = pd.DataFrame(columns=cols, data=test_feature[0])
test_feature['label'] = test_y
test_feature.to_csv('/home/meng/PyProject/Motor-life-prediction/feature_data/test.csv', index=False)

train_feature = pd.DataFrame(columns=cols, data=train_feature[0])
train_feature['label'] = label
train_feature.to_csv("/home/meng/PyProject/Motor-life-prediction/feature_data/train.csv", index=False)
