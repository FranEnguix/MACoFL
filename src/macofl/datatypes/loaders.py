from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class DataLoaders:
    train_data: DataLoader
    validation_data: DataLoader
    test_data: DataLoader
