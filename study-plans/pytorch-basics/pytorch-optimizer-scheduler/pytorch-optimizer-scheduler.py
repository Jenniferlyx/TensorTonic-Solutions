import torch
import torch.nn as nn

def train_with_scheduler(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR, num_epochs: int) -> dict:
    """
    Returns losses and lrs as lists of Python floats in a dictionary.
    """
    losses = []
    lrs = []
    for epoch in range(num_epochs):
        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        total_loss = 0.0
        num_batches = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        losses.append(total_loss / num_batches)
        scheduler.step()
    return {"losses": losses, "lrs": lrs}

