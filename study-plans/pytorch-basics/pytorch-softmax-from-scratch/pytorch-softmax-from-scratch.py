import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    eps=1e-8
    max = torch.amax(logits, dim=-1, keepdim=True)
    return torch.exp(logits - max) / (torch.sum(torch.exp(logits - max), dim=-1, keepdim=True) + eps)
