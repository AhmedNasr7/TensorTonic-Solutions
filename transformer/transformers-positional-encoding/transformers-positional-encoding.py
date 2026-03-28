import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    indices = np.arange(seq_length).reshape(-1, 1)
    dim_pairs = np.arange(0, d_model, 2)
    omega = 10_000 ** (dim_pairs / d_model)
    angles = indices/omega
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    return pe 

    
    