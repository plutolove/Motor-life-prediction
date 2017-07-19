import os
import pandas
import numpy
from sklearn import preprocessing
from keras.models import model_from_json
import keras.backend as K

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
col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 
'Voltage 2', 'Voltage 3', 'Accelerometer 1',
       'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']

data, label = load_data(col)

model = model_from_json(open('model.json').read())
model.load_weights('model.h5')
get_feature = K.function([model.layers[0].input, K.learning_phase()], [model.layers[9].output])
data = data.reshape(data.shape[0], 200, 13, 1)
feature = get_feature([data, 5])

print numpy.shape(feature)