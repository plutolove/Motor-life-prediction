import os
import pandas
import numpy
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import utils
from Load_Data import load_test, load_train, shuffle_data

def get_CNN():
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
    model.add(Dense(280, activation="relu"))
    model.add(Dense(28, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    return model


col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 
'Voltage 2', 'Voltage 3', 'Accelerometer 1',
       'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']

data, label = load_train(col, '/media/meng/9079-7B0D/clean_data/train/', 6)
data, label = shuffle_data(data, label)
test, test_y = load_test(col, '/media/meng/9079-7B0D/clean_data/test/')
data = data.reshape(data.shape[0], 200, 13, 1)
test = test.reshape(test.shape[0], 200, 13, 1)
test_y = utils.to_categorical(test_y, num_classes=28)
label = utils.to_categorical(label, num_classes=28)
model = get_CNN()
model.fit(data, label, batch_size=200, nb_epoch=5,
          verbose=1, shuffle=True, validation_data=(test, test_y))

json_string = model.to_json()
open('/home/meng/PyProject/Motor-life-prediction/classify/model.json','w').write(json_string) 
model.save_weights('/home/meng/PyProject/Motor-life-prediction/classify/model.h5')
