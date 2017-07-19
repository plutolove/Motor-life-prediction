import os
import pandas
import numpy
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import utils
from keras.models import model_from_json
import keras.backend as K

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

def get_model():
    model = Sequential()
    model.add(Conv2D(7, (2, 2), activation="relu", input_shape=(200, 13, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(9, (2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #model.add(Conv2D(11, (2, 2), activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(360, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(180, activation="relu"))
    model.add(Dense(18, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    return model


col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 
'Voltage 2', 'Voltage 3', 'Accelerometer 1',
       'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']

data, label = load_data(col)
data, label = shuffle_data(data, label)
data = data.reshape(data.shape[0], 200, 13, 1)

label = utils.to_categorical(label, num_classes=18)

model = get_model()
model.fit(data, label, batch_size=100, nb_epoch=5,
          verbose=1, shuffle=True, validation_split=0.3)

json_string = model.to_json()
open('model.json','w').write(json_string) 
model.save_weights("model.h5")
