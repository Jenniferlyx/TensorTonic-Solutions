import torch

class TransformPipeline:
    def __init__(self, mean: list, std: list):
        # mean/std: 长度为 C 的列表
        # tensor: (C,) → view: (C, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns a normalized float32 tensor with shape (C, H, W).
        """
        x = image.float() / 255.0
        x = x.permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x
