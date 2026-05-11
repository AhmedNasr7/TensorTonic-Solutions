import torch
import math

def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float64)
    return x.to(dtype=torch.float64)


def scale_residual_weights(W, N):
    """
    Returns: nested list of scaled weights, rounded to 4 decimals.
    """
    W = to_tensor(W)

    W_scaled = W / math.sqrt(N)

    W_scaled = torch.round(W_scaled * 10000) / 10000

    return W_scaled.tolist()


def forward_with_scaling(x, weights_list, N, use_scaling):
    """
    Returns: L2 norm of final activation as float, rounded to 4 decimals.
    """
    x = to_tensor(x)

    for W in weights_list:
        W = to_tensor(W)

        if use_scaling:
            W = W / math.sqrt(N)

        x = x + W @ x

    norm = torch.norm(x, p=2).item()

    return round(norm, 4)