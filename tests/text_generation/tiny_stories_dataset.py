from torch.utils.data import IterableDataset
import torch

class TinyStoriesDataset(IterableDataset):
    def __init__(self, dataset: list, sequence_length: int):
        """
        Dataset itératif pour générer des séquences (entrée, cible).
        """
        self.dataset = dataset
        self.sequence_length = sequence_length

    def __iter__(self):
        """
        Générateur qui produit les couples (entrée, cible) séquentiellement.
        """
        for tokens in self.dataset:
            if len(tokens) > self.sequence_length:
                for i in range(len(tokens) - self.sequence_length):
                    X = tokens[i : i + self.sequence_length]
                    y = tokens[i + 1 : i + self.sequence_length + 1]
                    yield torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)