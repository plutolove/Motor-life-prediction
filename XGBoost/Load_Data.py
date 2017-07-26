import numpy as np
import pandas as pd

def load_data(path):
    data = pd.read_csv(path).values
    X = data[:,0:256]
    Y = data[:, 256]
    return X, Y