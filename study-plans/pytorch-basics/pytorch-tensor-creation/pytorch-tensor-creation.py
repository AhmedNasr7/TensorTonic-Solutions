import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method == "zeros":
        tensor = torch.zeros(shape) 
    if method == "ones":
        tensor = torch.ones(shape) 
    if method == "full":
        tensor = torch.full(shape, value)

    return tensor.tolist()