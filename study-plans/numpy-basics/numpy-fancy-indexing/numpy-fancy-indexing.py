import numpy as np

def select_by_index(arr: list, indices: list, axis: int) -> np.ndarray:
    """
    Returns a 2D float64 array of the selected rows or columns.
    """
    arr = np.array(arr, dtype=np.float64)
    indices = np.array(indices, dtype=int)
    # indexed = arr[indices, :]
    indexed = np.take(arr, indices, axis=axis)
    return indexed
