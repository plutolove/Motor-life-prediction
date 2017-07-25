import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle

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
                tmp = np.reshape(tmp, 200*13, -1)
                ret.append(tmp)
                label.append(lab)
        if types < len(dicit):
            types = len(dicit)
        num_files = num_files + len(filelist)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    ret = min_max_scaler.fit_transform(ret)
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
                tmp = np.reshape(tmp, 200*13, -1)
                ret.append(tmp)
                label.append(lab)
        if types < len(dicit):
            types = len(dicit)
        num_files = num_files + len(filelist)
    min_max_scaler = preprocessing.MinMaxScaler()
    ret = min_max_scaler.fit_transform(ret)
    label = np.array(label)
    return ret, label

'''
col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 
'Voltage 2', 'Voltage 3', 'Accelerometer 1',
       'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
train, label, types = load_train(col, '/media/meng/9079-7B0D/clean_data/train/')
print np.shape(train)
print np.shape(label)
print types
'''

def shuffle_data(data, label):
    X, Y = shuffle(data, label, random_state=0)
    return X, Y
