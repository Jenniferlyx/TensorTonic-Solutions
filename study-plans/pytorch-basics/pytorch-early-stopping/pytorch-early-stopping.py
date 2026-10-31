import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for x, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_samples += 1
        train_losses.append(train_loss / train_samples)

        model.eval()
        total_val = 0.0
        n_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val += loss.item()
                n_val += 1
        val_losses.append(total_val / n_val)
        # Early stopping check
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                return {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "stopped_epoch": epoch,
                }
    return {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "stopped_epoch": max_epochs,
            }
