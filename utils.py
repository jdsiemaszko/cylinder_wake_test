import numpy as np
import pandas as pd


def unpack_data(source):
    data = np.array(pd.read_csv(source, sep=' ', header=None))
    x = data[:, 0]
    y = data[:, 1]
    g = data[:, 2]
    sigma = np.full(len(g), 0.01)
    return x, y, g, sigma