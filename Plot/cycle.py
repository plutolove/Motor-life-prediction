import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.random as random
import math


def load_test(col, path):
    print 'load test data from '+path
    dirlist = os.listdir(path)
    ret = []
    label = []
    for dirname in dirlist:
        filelist = os.listdir(path + dirname)
        dicit = set()
        print dirname
        tmp = []
        for filename in filelist:
            fullpath = path + dirname + '/'+filename
            data = pd.read_csv(fullpath)
            data = data.loc[:, col].values
            data = data[:, 0]
            fi = filename[0:15]

            if len(dicit) == 0:
                dicit.add(fi)

            if fi in dicit:
                mid = random.choice(data, 1000, replace=False)
                ret.append(mid)
            else:
                #ret.append(tmp)
                #tmp = []
                dicit.add(fi)
    return ret

def mean_variance(path, col):
    filelist = os.listdir(path)
    mean = []
    var = []
    for filename in filelist:
        fullpath = path + filename
        data = pd.read_csv(fullpath)
        data = data.loc[:, col].values
        single_mean = []
        single_var = []
        for i in range(len(col)):
            single_mean.append(np.mean(data[:, i]))
            single_var.append(np.var(data[:, i]))

        mean.append(single_mean)
        var.append(single_var)
    return np.array(mean), np.array(var)

def Plot(mean, var, col):
    n = len(col)
    styles = ['g-', 'b-']
    X = list(xrange(np.shape(mean)[0]))    
    for i in range(n):
        with sns.axes_style('darkgrid'):
            fig = plt.figure()
            plt.subplot(121)
            Y = mean[:, i]
            plt.plot(X, Y, styles[0])
            plt.title('Mean of ' + col[i])
            plt.xlabel('files')
            plt.ylabel('mean')

            plt.subplot(122)
            Y = var[:, i]
            plt.plot(X, Y, styles[0], linewidth=2)
            plt.title('Variance of ' + col[i])
            plt.xlabel('files')
            plt.ylabel('variance')

            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.45)
            #plt.legend()
            plt.show()
            plt.savefig('/home/meng/PyProject/Motor-life-prediction/Plot/pic/motor7/'+str(i)+'.png', dpi=fig.dpi)

def plot_cycle(data, title):
    rows = np.shape(data)[0]
    plt.figure()
    styles = ['g-', 'b-']
    for i in range(rows):
        with sns.axes_style('darkgrid'):
            Y = data[i]
            X = list(xrange(i*1000, (i+1)*1000))
            plt.plot(X, Y, styles[i%2])
            plt.title(title)
            plt.xlabel('samples')
            plt.ylabel('values')
    #plt.show()
    plt.savefig('/home/meng/PyProject/Motor-life-prediction/Plot/cycle/motor7/'+title+'.png')

def plot_(path, col):
    filelist = os.listdir(path)
    for filename in filelist:
        fullpath = path + filename
        data = pd.read_csv(fullpath)
        data = data.loc[:, col].values
        data = data[:, 0]
        X = list(xrange(len(data)))
        plt.figure()
        with sns.axes_style('darkgrid'):
            plt.plot(X, data)
        plt.show()

col = ['X_Value', 'Current 1', 'Current 2', 'Current 3', 'Voltage 1', 'Voltage 2', 'Voltage 3', 'Accelerometer 1', 'Accelerometer 2', 'Microphone', 'Tachometer', 'Temperature', 'Output Current', 'Output Voltage']
#x = load_test(['Current 3'], '/media/meng/9079-7B0D/clean_data/test/')
def run():
    for item in col:
        print item
        x = load_test([item], '/media/meng/9079-7B0D/clean_data/test/')
        print np.shape(x)
        plot_cycle(x, item)
#plot_('/media/meng/9079-7B0D/clean_data/train/clean_m7/', ['Current 1'])


def get_err_file(col, path, delt):
    print 'load test data from '+path
    dirlist = os.listdir(path)
    ret = []
    for dirname in dirlist:
        filelist = os.listdir(path + dirname)
        for filename in filelist:
            fullpath = path + dirname + '/'+filename
            data = pd.read_csv(fullpath)
            data = data.loc[:, col].values
            data = data[:, 0]
            
            rate = 0.0

            for item in data:
                if math.fabs(item) < delt:
                    rate = rate + 1.0
            
            if rate / len(data) > 0.5:
#                print fullpath
                ret.append(fullpath)
    return ret

def clean_data(filelist, clean_file, col):
    clean = pd.read_csv(clean_file)
    clean_col = clean[col].values
    for fname in filelist:
        data = pd.read_csv(fname)
        data[col] = clean_col
        data.to_csv(fname, index=False)

def run_clean(col, delt):
    list_of_error = get_err_file(col, '/media/meng/9079-7B0D/clean_data/test/', delt)
    print list_of_error
    clean_data(list_of_error, '/media/meng/9079-7B0D/clean_data/test/clean_m7/SSrunLo20110613141950.csv', col[0])
cols = ['Voltage 3']
delt = 3
run_clean(cols, delt)
#run()
