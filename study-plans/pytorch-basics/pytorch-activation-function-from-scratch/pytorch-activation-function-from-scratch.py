import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(data=x, dtype=torch.float32)
        
        
    if method == "relu":
        act = torch.clamp(x, min=0)

    elif method == "sigmoid":
        act = 1. / (1. + torch.exp(-x))

    elif method == "tanh":
        act = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
     
    elif method == "leaky_relu":
        act = torch.where(x > 0, x, 0.01 * x)
        


    return act.tolist()
