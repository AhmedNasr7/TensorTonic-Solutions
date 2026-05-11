import numpy as np

def to_np_array(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    # numpy li_norm: np.linalg.norm(x, ord=i)


    v = to_np_array(v)
    
    l1_norm = np.sum(np.absolute(v))
    l2_norm = np.sqrt(np.sum((v ** 2)))
    l_inf_norm = np.max(np.absolute(v), axis=-1)

    norms = np.array([l1_norm, l2_norm, l_inf_norm], dtype=np.float64)

    return norms