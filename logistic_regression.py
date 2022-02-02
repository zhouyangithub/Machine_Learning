import numpy as np
def sigmod(x):
    z = 1/(1+np.exp(-x))
    return z
