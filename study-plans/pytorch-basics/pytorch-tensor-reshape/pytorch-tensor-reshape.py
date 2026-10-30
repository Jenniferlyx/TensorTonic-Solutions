import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    if op == "flatten":
        return x.flatten()
    elif op == "squeeze":
        return x.squeeze()
    elif op == "unsqueeze":
        return x.unsqueeze()
    elif op == "transpose":
        return x.T
