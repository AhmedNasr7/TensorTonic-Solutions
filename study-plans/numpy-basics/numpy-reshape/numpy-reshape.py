import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    
    arr = np.array(data, dtype=np.float64)


    if operation == "flatten":
        return arr.flatten()
    elif operation == "transpose":
        return arr.T
    elif operation == "add_batch":
        return arr.reshape(1, *arr.shape)

        
