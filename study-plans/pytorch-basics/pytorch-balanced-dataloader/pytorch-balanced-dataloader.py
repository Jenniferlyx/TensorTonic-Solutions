import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader with inverse-frequency sampling and replacement.
    """
    class_count = torch.bincount(labels)
    class_weights = 1.0 / class_count.float()
    # torch.stack([class_weights[i] for i in labels])
    sample_weights = class_weights[labels] 
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    

