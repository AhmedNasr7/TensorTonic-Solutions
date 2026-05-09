import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(data=values, dtype=torch.float32, requires_grad=True)

    y = torch.sum(values ** 3 + 2 * values)
    y.backward()
    grad = values.grad

    return grad.tolist()

    