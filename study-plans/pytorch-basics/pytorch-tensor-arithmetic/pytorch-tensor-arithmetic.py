import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(data=x, dtype=torch.float32)
        
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(data=y, dtype=torch.float32)
        
    #add, multiply, matmul, power, and max.
    if op == "add":
        result = torch.add(x, y)
    
    if op == "multiply":
        result = torch.mul(x, y)
        
    if op == "matmul":
        result = torch.matmul(x, y)

    if op == "power":
        result = torch.pow(x, y)

    if op == "max":
        result = torch.max(x, y)

    result = result.tolist()

    return result