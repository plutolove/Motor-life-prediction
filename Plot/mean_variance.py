import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def mean_variance_cycle(path, col):
    filelist = os.listdir(path)
    mean = []
    var = []
    s = set()

    dataset = []

    for filename in filelist:
        fullpath = path + filename
        fi = filename[0:15]
        if len(s) == 0:
            s.add(fi)
        data = pd.read_csv(fullpath)
        data = data.loc[:, col].values

        single_mean = []
        single_var = []

        if fi in s:
            n = np.shape(data)[0]
            for i in range(n):
                dataset.append(data[i])
        else:
            s.add(fi)
            dataset = np.array(dataset)
            for i in range(len(col)):
                single_mean.append(np.mean(dataset[:, i]))
                single_var.append(np.var(dataset[:, i]))
            dataset = []
            n = np.shape(data)[0]
            for i in range(n):
                dataset.append(data[i])

        if len(single_mean) > 0:
            var.append(single_var)
            mean.append(single_mean)
    
    single_mean = []
    single_var = []
    dataset = np.array(dataset)
    for i in range(len(col)):
        single_mean.append(np.mean(dataset[:, i]))
        single_var.append(np.var(dataset[:, i]))
    var.append(single_var)
    mean.append(single_mean)
    return np.array(mean), np.array(var)



def Plot(mean, var, col, path):
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
            #plt.show()
            plt.savefig(path+col[i]+'.png', dpi=fig.dpi)


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
mean, var = mean_variance_cycle('/media/meng/9079-7B0D/clean_data/clean/clean_m7/', col)
Plot(mean, var, col, '/home/meng/PyProject/Motor-life-prediction/Plot/pic/mean_var_cycle/motor7/')
