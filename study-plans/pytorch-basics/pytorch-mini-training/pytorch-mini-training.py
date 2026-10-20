import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    model.train()
    
    running_loss = 0.
    total_loss = 0.
    for i, data in enumerate(dataloader):
        x, y = data
        optimizer.zero_grad()
        preds = model.forward(x)
        loss = criterion(y, preds)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_loss = running_loss / (i + 1)

    

    return total_loss
        
