import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mean = torch.mean(X, dim=0, keepdim=True, dtype=torch.float32)
    var =torch.var(X, dim=0, keepdim=True, unbiased=False)

    Y = gamma * ((X - mean) / torch.sqrt(var + eps)) + beta

    return Y
