import numpy as np

def to_np_array(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x



def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = to_np_array(a)
    b = to_np_array(b)
    eps = 1e-18

    cos_similarity = np.dot(a, b) / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2) + eps)

    return cos_similarity
    