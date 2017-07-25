import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import utils
from keras.callbacks import EarlyStopping

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

def shuffle_data(data, label):
    X, Y = shuffle(data, label, random_state=0)
    return X, Y
'''

def load_train(col, path, num_train):
    print 'load train data from '+path
    dirlist = os.listdir(path)
    dirlist.sort()
    ret = []
    label = []
    types = 0
    num_files = 0
    for i in range(num_train):
        dirname = dirlist[i]
        print dirname
        filelist = os.listdir(path + dirname)
        dicit = set()
        for filename in filelist:
            fullpath = path + dirname + '/'+filename
            #print fullpath
            data = pd.read_csv(fullpath)
            data = data.loc[:, col].values
            rows = np.shape(data)[0]
            dicit.add(filename[0:15])
            lab = len(dicit) - 1
            for i in range(0, rows, 200):
                tmp = []
                for j in range(i, i+200):
                    tmp.append(data[j, :])
                ret.append(tmp)
                label.append(lab)
        if types < len(dicit):
            types = len(dicit)
        num_files = num_files + len(filelist)
    
    ret = np.asarray(ret, dtype="float32")
    ret = ret.reshape(ret.shape[0], -1)
    #ret = np.reshape(ret, (num_files * 100, -1))
    #min_max_scaler = preprocessing.MinMaxScaler()
    #ret = min_max_scaler.fit_transform(ret)
    #ret = np.reshape(ret, (num_files * 100, 200, -1))
    ret = ret.reshape(ret.shape[0], 200, -1)
    label = np.array(label)
    return ret, label


def load_test(col, path):
    print 'load test data from '+path
    dirlist = os.listdir(path)
    ret = []
    label = []
    types = 0
    num_files = 0
    for dirname in dirlist:
        filelist = os.listdir(path + dirname)
        dicit = set()
        print dirname
        for filename in filelist:
            fullpath = path + dirname + '/'+filename
            #print fullpath
            data = pd.read_csv(fullpath)
            data = data.loc[:, col].values
            rows = np.shape(data)[0]
            dicit.add(filename[0:15])
            lab = len(dicit) - 1
            for i in range(0, rows, 200):
                tmp = []
                for j in range(i, i+200):
                    tmp.append(data[j, :])
                ret.append(tmp)
                label.append(lab)
        if types < len(dicit):
            types = len(dicit)
        num_files = num_files + len(filelist)
    ret = np.asarray(ret, dtype="float32")
    ret = np.reshape(ret, (num_files * 100, -1))
    #min_max_scaler = preprocessing.MinMaxScaler()
    #ret = min_max_scaler.fit_transform(ret)
    ret = np.reshape(ret, (num_files * 100, 200, -1))
    label = np.array(label)
    return ret, label



def shuffle_data(data, label):
    X, Y = shuffle(data, label, random_state=0)
    return X, Y



def get_LSTM():
    model = Sequential()
    model.add(ConvLSTM2D(filters=10, kernel_size=(2, 2), input_shape=(200, 13, 1, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=15, kernel_size=(2, 2), padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(54, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_absolute_error", optimizer='rmsprop', metrics=['accuracy'])
    return model

'''
col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 
'Voltage 2', 'Voltage 3', 'Accelerometer 1',
       'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']

data, label = load_data(col)
#data, label = shuffle_data(data, label)
data = data.reshape(data.shape[0], 1, 200, 13, 1)
model = get_model()
model.fit(data, label, batch_size=100, nb_epoch=15,
          verbose=1, shuffle=False, validation_split=0.1)
'''
def run():
    col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 'Voltage 2', 'Voltage 3', 'Accelerometer 1', 'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
    #train, label = load_train(col, '/media/meng/9079-7B0D/clean_data/train/', 1)
    test, test_y = load_test(col, "/media/meng/9079-7B0D/clean_data/test/")
    #train, label = shuffle_data(train, label)
    #label = label / 28.0
    test_y = test_y
    test = test.reshape(test.shape[0], 200, 13, 1, 1)

    train, label = load_train(col, '/media/meng/9079-7B0D/clean_data/train/', 3)
    train, label = shuffle_data(train, label)
    label = label
    train = train.reshape(train.shape[0], 200, 13, 1, 1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    model = get_LSTM()
    print '---------------training model-------------------'
    model.fit(train, label, batch_size=200, epochs=50, callbacks=[early_stopping], verbose=1, shuffle=True, validation_data=(test, test_y))

    json_string = model.to_json()
    open('/home/meng/PyProject/Motor-life-prediction/lstm_model/model.json','w').write(json_string) 
    model.save_weights('/home/meng/PyProject/Motor-life-prediction/lstm_model/model.h5')

run()
