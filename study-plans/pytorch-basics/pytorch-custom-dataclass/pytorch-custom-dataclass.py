import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, data: list, label_col: int):
        t = torch.tensor(data, dtype=torch.float32)
        self.labels = t[:, label_col].unsqueeze(1)
        feat_cols = [i for i in range(t.shape[1]) if i != label_col]
        self.features = t[:, feat_cols]

    def __len__(self) -> int:
        """
        Returns the number of rows.
        """
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, label) as float32 tensors of shapes (D,) and (1,).
        """
        return self.features[idx], self.labels[idx]
