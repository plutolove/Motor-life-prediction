import os
import pandas
import numpy
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

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

'''
col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 
'Voltage 2', 'Voltage 3', 'Accelerometer 1',
       'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']

data, label = load_data(col)
data, label = shuffle_data(data, label)
data = data.reshape(data.shape[0], 200, 13, 1)
label = utils.to_categorical(label, num_classes=18)
model = get_model()
model.fit(data, label, batch_size=100, nb_epoch=300,
          verbose=1, shuffle=True, validation_split=0.3)
'''