import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method == "relu":
        return torch.clamp(x, min=0)
    elif method == "sigmoid":
        return 1 / (1 + torch.exp(-x))
    elif method == "tanh":
        # return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        # numerical stable version
        magnitude = torch.where(x >= 0, x, -x)
        decay = torch.exp(-2 * magnitude)
        positive = (1 - decay) / (1 + decay)
        return torch.where(x >= 0, positive, -positive)
    elif method == "leaky_relu":
        return torch.where(x > 0, x, 0.01 * x)
