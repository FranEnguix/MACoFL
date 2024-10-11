from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
