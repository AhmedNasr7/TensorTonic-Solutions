import numpy as np

def create_sequence(start, stop, param, kind):
    """
    Returns: 1D ndarray of float64 values
    """
    if kind == 'arange':
        return np.arange(start, stop, step=param, dtype=np.float64)

    elif kind == "linspace":
        return np.linspace(start, stop, num=param, dtype=np.float64)
        
