import numpy as np


def to_np_array(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x, np.float64)
    return x

    
def linear_combination(vectors, coefficients):
    """
    Returns: float64 array, the weighted sum of vectors.
    """

    vectors = to_np_array(vectors)
    coefficients = to_np_array(coefficients)

    w = []

    w = np.matmul(vectors.T, coefficients)


    return w
        

    