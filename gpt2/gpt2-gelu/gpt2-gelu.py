import torch

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor with GELU applied element-wise
    """
    return x * 0.5 * (1 + torch.erf(x / 2 ** 0.5))