import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    if op == "add":
        return x + y
    elif op == "multiply":
        return x * y
    elif op == "matmul":
        return x @ y
    elif op == "power":
        return x ** y
    elif op == "max":
        return torch.maximum(x, y)
