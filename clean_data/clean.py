import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle

def load_train(col, path, num_train):
    print 'load train data from '+path
    dirlist = os.listdir(path)
    dirlist.sort()
    for i in range(num_train):
        dirname = dirlist[i]
        #print dirname
        filelist = os.listdir(path + dirname)
        for filename in filelist:
            fullpath = path + dirname + '/'+filename
            data = pd.read_csv(fullpath)
            data = data.loc[:, col].values
            rows = np.shape(data)[0]
            for row in data:
                for num in row:
                    if len(str(num)) == 1:
                        print fullpath
                        print num


col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 
'Voltage 2', 'Voltage 3', 'Accelerometer 1',
       'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
load_train(col, '/media/meng/9079-7B0D/clean_data/train/', 6)
