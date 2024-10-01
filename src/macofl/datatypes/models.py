from dataclasses import dataclass
from typing import OrderedDict

import torch
from torch import Tensor, nn


@dataclass
class Model:
    """
    Handles the Neural Network model operations and weights.
    """

    model: nn.Module
    local_weights: OrderedDict[str, Tensor]
    other_weights: OrderedDict[str, Tensor]
    batch_size: int

    def save_model_weights_to_file(self, filepath: str) -> None:
        """
        Saves the model's weights to a file.

        Args:
            filepath (str): The path to the file where weights will be saved.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model_weights_from_file(self, filepath: str) -> None:
        """
        Loads weights from a file into the model.

        Args:
            filepath (str): The path to the file from which to load weights.
        """
        self.model.load_state_dict(torch.load(filepath))
