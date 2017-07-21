'''
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]

#motor1 test result
motor1_test_loss = [0.3029, 0.2937, 0.2675, 0.2266, 0.1824, 0.1735]
motor1_test_acc = [0.9589, 0.9591, 0.9643, 0.9648, 0.9659, 0.9638]

#motor2 test result
motor2_test_loss = [0.2549, 0.2097, 0.1990, 0.1957, 0.1876, 0.1953]
motor2_test_acc = [0.9578, 0.9643, 0.9642, 0.9643, 0.9627, 0.9519]

#motor3 test result
motor3_test_loss = [0.2645, 0.2254, 0.1863, 0.1685, 0.1585, 0.1575]
motor3_test_acc = [0.9643, 0.9639, 0.9654, 0.9657, 0.9653, 0.9631]

with sns.axes_style('darkgrid'):
    plt.subplot(321)
    plt.plot(x, motor1_test_acc)
    plt.title('acc of motor1 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('acc')

    plt.subplot(322)
    plt.plot(x, motor1_test_loss)
    plt.title('loss of motor1 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('loss')

    plt.subplot(323)
    plt.plot(x, motor2_test_acc)
    plt.title('acc of motor2 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('acc')

    plt.subplot(324)
    plt.plot(x, motor2_test_loss)
    plt.title('loss of motor2 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('loss')

    plt.subplot(325)
    plt.plot(x, motor3_test_acc)
    plt.title('acc of motor3 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('acc')

    plt.subplot(326)
    plt.plot(x, motor3_test_loss)
    plt.title('loss of motor3 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('loss')
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.45)
    plt.show()
'''
#x_train = np.linspace(-2*np.pi, 2*np.pi, 1000)  #array: [1000,]  
#x_train = np.array(x_train).reshape((len(x_train), 1)) #reshape to matrix with [100,1]
#n=0.1*np.random.rand(len(x_train),1) #generate a matrix with size [len(x),1], value in (0,1),array: [1000,1]  
#y_train=np.sin(x_train)+n
#print y_train

'''
sns.set(style="darkgrid", color_codes=True)
data = pd.DataFrame({'x': x, 'y': motor2_test_acc})
plt.subplot(211)
ax = sns.factorplot(x='x', y='y', truncate=True, size=5, data=data)
plt.subplot(212)
ax1 = sns.factorplot(x='x', y='y', truncate=True, size=5, data=data)
plt.show()
'''

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
    model.add(Dense(1))#, activation="sigmoid"))
    model.compile(loss="mse", optimizer='rmsprop', metrics=['accuracy'])
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
    plt.show()
    savefig('test_motor3.png')

    '''
    with sns.axes_style('darkgrid'):
        plt.subplot(211)
        plt.plot(epoch, loss, "g-",label='train 1')
        plt.title('loss of training')
        plt.xlabel('epochs')
        plt.ylabel('loss')

        plt.subplot(212)
        plt.plot(epoch, val_loss)
        plt.title('val_loss of training')
        plt.xlabel('epochs')
        plt.ylabel('val_loss')

        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.45)
        plt.show()
    '''

hists = []

col = ['Current 1', 'Current 2', 'Current 3', 'Voltage 1', 'Voltage 2', 'Voltage 3', 'Accelerometer 1', 'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
#train, label = load_train(col, '/media/meng/9079-7B0D/clean_data/train/', 1)
test, test_y = load_test(col, "/media/meng/9079-7B0D/clean_data/test/")
#train, label = shuffle_data(train, label)
#label = label / 28.0
test_y = test_y / 28.0
#train = train.reshape(train.shape[0], 200, 13, 1)
test = test.reshape(test.shape[0], 200, 13, 1)
model = get_CNN()
early_stopping = EarlyStopping(monitor='val_loss', patience=100)

for i in range(6):
    train, label = load_train(col, '/media/meng/9079-7B0D/clean_data/train/', i)
    train, label = shuffle_data(train, label)
    label = label / 28.0
    train = train.reshape(train.shape[0], 200, 13, 1)
    hist = model.fit(train, label, batch_size=200, epochs=50, callbacks=[early_stopping], verbose=1, shuffle=True, validation_data=(test, test_y))
    hists.append(hist)

figure_out(hists)


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

