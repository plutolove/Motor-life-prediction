from LoadData import load_test, load_train, shuffle_data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def get_autocode():
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(2600, )))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2600, activation='sigmoid'))
    model.compile(loss="mean_absolute_error", optimizer='rmsprop', metrics=['accuracy'])
    return model

col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 'Voltage 2', 'Voltage 3', 'Accelerometer 1', 'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
path = '/media/meng/9079-7B0D/clean_data/clean/'
data, label = load_train(col, path+'train/', 6)
test_x, test_y = load_test(col, path+'test/')
print np.shape(data)
data, label = shuffle_data(data, label)
model = get_autocode()
model.fit(data, data, verbose=1, epochs=50, batch_size=200, validation_data=(test_x, test_x))
json_string = model.to_json()
open('/home/meng/PyProject/Motor-life-prediction/autoencoder/model.json','w').write(json_string) 
model.save_weights('/home/meng/PyProject/Motor-life-prediction/autoencoder/model.h5')
