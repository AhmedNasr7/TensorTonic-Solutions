import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(data=x, dtype=torch.float32)

    if op == "flatten":
        result = torch.flatten(x)
    
    if op == "squeeze":
        result = torch.squeeze(x)
        
    if op == "transpose":
        result = x.T

    result = result.tolist()

    return result
