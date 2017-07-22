import os
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig

def load_train(col, path, num_train):
    print 'load train data from '+path
    dirlist = os.listdir(path)
    dirlist.sort()
    ret = []
    label = []
    types = 0
    num_files = 0
    #for i in range(num_train):
    dirname = dirlist[num_train]
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
    min_max_scaler = preprocessing.MinMaxScaler()
    ret = min_max_scaler.fit_transform(ret)
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
    min_max_scaler = preprocessing.MinMaxScaler()
    ret = min_max_scaler.fit_transform(ret)
    ret = np.reshape(ret, (num_files * 100, 200, -1))
    label = np.array(label)
    return ret, label



def shuffle_data(data, label):
    X, Y = shuffle(data, label, random_state=0)
    return X, Y


def get_CNN():
    model = Sequential()
    model.add(Conv2D(11, (2, 2), activation="relu", input_shape=(200, 13, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(13, (2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #model.add(Conv2D(11, (2, 2), activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(360, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(280, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_absolute_error", optimizer='rmsprop', metrics=['accuracy'])
    return model

def figure_out(history):
    
    epoch    = history[0].epoch
    #acc      = hist['acc']
    #val_acc  = hist['val_acc']
    #print epoch, loss, val_loss
    styles = ['g-', 'b-', 'r-', 'c-', 'm-', 'y-']
    n = len(history)
    with sns.axes_style('darkgrid'):
        plt.subplot(211)
        for i in range(n):
            hist = history[i].history
            loss     = hist['loss']
            plt.plot(epoch, loss, styles[i], label='train '+str(i)+' motors', linewidth=2)
            plt.title('loss of training')
            plt.xlabel('epochs')
            plt.ylabel('loss')
        
        plt.legend()

        plt.subplot(212)
        for i in range(n):
            hist = history[i].history
            loss     = hist['val_loss']
            plt.plot(epoch, loss, styles[i], label='train '+str(i)+' motors', linewidth=2)
            plt.title('loss of training')
            plt.xlabel('epochs')
            plt.ylabel('val_loss')
        
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.45)
    plt.legend()
    #plt.show()
    savefig('cnn_line.png')

def run_cnn_linear():
    hists = []
    col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 'Voltage 2', 'Voltage 3', 'Accelerometer 1', 'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
    #train, label = load_train(col, '/media/meng/9079-7B0D/clean_data/train/', 1)
    test, test_y = load_test(col, "/media/meng/9079-7B0D/clean_data/test/")
    #train, label = shuffle_data(train, label)
    #label = label / 28.0
    test_y = test_y
    #train = train.reshape(train.shape[0], 200, 13, 1)
    test = test.reshape(test.shape[0], 200, 13, 1)
    model = get_CNN()
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)

    for i in range(6):
        train, label = load_train(col, '/media/meng/9079-7B0D/clean_data/train/', i)
        train, label = shuffle_data(train, label)
        label = label
        train = train.reshape(train.shape[0], 200, 13, 1)
        hist = model.fit(train, label, batch_size=200, epochs=40, callbacks=[early_stopping], verbose=1, shuffle=True, validation_data=(test, test_y))
        hists.append(hist)
        json_string = model.to_json()
        open('/home/meng/PyProject/Motor-life-prediction/cnn_line_model/std_model_'+str(i)+'.json','w').write(json_string) 
        model.save_weights('/home/meng/PyProject/Motor-life-prediction/cnn_line_model/std_model_'+str(i)+'.h5')
    figure_out(hists)
