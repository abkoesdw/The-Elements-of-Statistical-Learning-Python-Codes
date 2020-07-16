import numpy as np


def scale(x):
    mean = x.values.mean(axis=0, keepdims=True)
    stdev = x.values.std(axis=0, keepdims=True, ddof=0)
    return np.divide(np.subtract(x, mean), stdev)
