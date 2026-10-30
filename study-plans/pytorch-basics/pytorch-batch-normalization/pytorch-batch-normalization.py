import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mean_x = X.mean(dim=0)
    var_x = ((X - mean_x) ** 2).mean(dim=0)
    return gamma * (X - mean_x) / (var_x + eps) ** 0.5 + beta
