import torch

def initialize_weights(fan_in: int, fan_out: int, method: str) -> torch.Tensor:
    """
    Returns a float32 weight tensor with shape (fan_out, fan_in).
    """
    if method == "xavier_uniform":
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return torch.empty(fan_out, fan_in, dtype=torch.float32).uniform_(-limit, limit)
    elif method == "xavier_normal":
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return torch.empty(fan_out, fan_in, dtype=torch.float32).normal_(0.0, std)
    elif method == "he_uniform":
        limit = math.sqrt(6.0 / fan_in)
        return torch.empty(fan_out, fan_in, dtype=torch.float32).uniform_(-limit, limit)
    elif method == "he_normal":
        std = math.sqrt(2.0 / fan_in)
        return torch.empty(fan_out, fan_in, dtype=torch.float32).normal_(0.0, std)
