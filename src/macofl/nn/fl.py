from typing import Optional

import torch

from ..datatypes import DataLoaders, Model


class FederatedLearning:

    def __init__(self, device: Optional[str] = None) -> None:
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.dataloaders: DataLoaders
        self.model: Model
