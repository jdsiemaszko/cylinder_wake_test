import numpy as np
import pandas as pd


def unpack_data(source, core_size=0.01):
    data = np.array(pd.read_csv(source, sep=' ', header=None))
    x = data[:, 0]
    y = data[:, 1]
    g = data[:, 2]
    sigma = np.full(len(g), core_size)
    return x, y, g, sigma

def create_linear(x0, y0, x1, y1):
    if x0 == x1:
        raise ValueError('two reference x values cannot be the same!')
    
    a = (y1 - y0) / (x1 - x0)
    b = y0 - a*x0
    return lambda index, x, y, g, sigma: a*x[index].mean()+b