import numpy as np


def to_np_array(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


    
def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = to_np_array(x)
    y = to_np_array(y)
    
    return np.sqrt(np.sum((x - y) ** 2))