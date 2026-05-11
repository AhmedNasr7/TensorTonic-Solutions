import numpy as np


def to_np_array(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x
    
def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x = to_np_array(x)
    y = to_np_array(y)
    
    return np.dot(x, y)