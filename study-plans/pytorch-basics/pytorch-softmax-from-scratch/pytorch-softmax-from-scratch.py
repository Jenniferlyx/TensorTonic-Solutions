import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    shifted = logits - torch.max(logits, dim=-1, keepdim=True).values
    exp_x = torch.exp(shifted)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
