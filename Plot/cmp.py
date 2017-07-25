import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path):
    data = pd.read_csv(path).values
    return data[:, 0]

def Plot(cnn, tree, real):
    X = list(xrange(len(cnn)))
    styles = ['g-', 'b-', 'r-']
    plt.figure()
    with sns.axes_style('darkgrid'):
        plt.plot(X, cnn, styles[0], label='CNN')
        plt.plot(X, tree, styles[1], label='xgboost')
        plt.plot(X, real, styles[2], label='Label')
        plt.xlabel('id')
        plt.ylabel('life')
    plt.legend()
    plt.show()

cnn = load_data('/home/meng/PyProject/Motor-life-prediction/feature_data/CNN.csv')
tree = load_data('/home/meng/PyProject/Motor-life-prediction/feature_data/xgboost.csv')
label = load_data('/home/meng/PyProject/Motor-life-prediction/feature_data/label.csv')

Plot(cnn, tree, label)
