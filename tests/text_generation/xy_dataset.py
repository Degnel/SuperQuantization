from torch.utils.data import Dataset
import torch

class XyDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Custom dataset to pair inputs X with targets y.
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]