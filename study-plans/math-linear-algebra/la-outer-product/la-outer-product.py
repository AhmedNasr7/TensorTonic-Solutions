import numpy as np

def to_np_array(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x, np.float64)
    return x

    
def outer_product(u, v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    u = to_np_array(u)
    v = to_np_array(v)

    print(u.shape, v.shape)

    if u.ndim == 1:
        u = u[:, np.newaxis]
    if v.ndim == 1:
        v = v[np.newaxis, :]



    return np.matmul(u, v)