import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """
    preds = model.forward(X)
    loss = criterion(preds, y)
    loss.backward()
    for param in model.parameters():
        if param.requires_grad == True:
            with torch.no_grad(): ## crucial for in-place parameter updates.
                param -= lr * param.grad

    model.zero_grad()

    return loss.item()
    
