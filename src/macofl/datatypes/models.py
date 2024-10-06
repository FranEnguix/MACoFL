import codecs
import copy
import pickle
from typing import Optional, OrderedDict

import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..datatypes.loaders import DataLoaders
from ..datatypes.metrics import ModelMetrics


class ModelManager:
    """
    Handles the Neural Network model training, validation and testing.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: _Loss,
        optimizer: Optimizer,
        batch_size: int,
        training_epochs: int,
        dataloaders: DataLoaders,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.dataloaders = dataloaders
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.initial_state: OrderedDict[str, Tensor] = copy.deepcopy(model.state_dict())
        self.pretrain_state: OrderedDict[str, Tensor] = copy.deepcopy(
            self.model.state_dict()
        )
        self.__training: bool = False

    def is_training(self) -> bool:
        return self.__training

    def replace_weights_and_biases(
        self, new_weights_and_biases: OrderedDict[str, Tensor]
    ) -> None:
        self.model.load_state_dict(state_dict=new_weights_and_biases)

    def train(self, epochs: Optional[int] = None) -> ModelMetrics:
        """
        Updates the model by training on the training dataset.
        """
        self.__training = True
        if epochs is None:
            epochs = self.training_epochs

        self.pretrain_state = copy.deepcopy(self.model.state_dict())

        # Training loop
        for _ in range(epochs):
            self.model.train()
            total_loss: float = 0.0
            correct: int = 0
            total_samples: int = 0

            images: Tensor
            labels: Tensor
            outputs: Tensor
            loss: Tensor
            predicted: Tensor

            for images, labels in self.dataloaders.train_data:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += int((predicted == labels).sum().item())

        self.__training = False
        accuracy: float = correct / total_samples
        resulting_loss: float = total_loss / len(self.dataloaders.train_data)
        metrics: ModelMetrics = ModelMetrics(accuracy=accuracy, loss=resulting_loss)
        return metrics

    def _inference(self, dataloader: DataLoader) -> ModelMetrics:
        """
        Performs inference on a given dataset and returns metrics.
        """

        # Validation
        self.model.eval()
        correct: int = 0
        total: int = 0
        predicted_labels: list[int] = []
        true_labels: list[int] = []
        total_loss: float = 0.0

        images: Tensor
        labels: Tensor
        outputs: Tensor
        loss: Tensor
        predicted: Tensor

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += int((predicted == labels).sum().item())
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy: float = correct / total
        resulting_loss: float = total_loss / len(dataloader)
        metrics: ModelMetrics = ModelMetrics(accuracy=accuracy, loss=resulting_loss)
        return metrics

    def inference(self) -> ModelMetrics:
        """
        Returns the TRAIN inference metrics.
        """
        return self._inference(dataloader=self.dataloaders.validation_data)

    def test_inference(self) -> ModelMetrics:
        """
        Returns the TEST inference metrics.
        """
        return self._inference(dataloader=self.dataloaders.test_data)

    @staticmethod
    def export_weights_and_biases(model: OrderedDict[str, Tensor]) -> str:
        return codecs.encode(pickle.dumps(model), encoding="base64").decode(
            encoding="utf-8"
        )

    @staticmethod
    def import_weights_and_biases(
        base64_codified_model: str,
    ) -> OrderedDict[str, Tensor]:
        return pickle.loads(
            codecs.decode(
                base64_codified_model.encode(encoding="utf-8"), encoding="base64"
            )
        )

    def save_model_to_file(self, filepath: str) -> None:
        """
        Saves the model into a file.

        Args:
            filepath (str): The path to the file where the model will be saved.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model_from_file(self, filepath: str) -> None:
        """
        Loads the model from a file.

        Args:
            filepath (str): The path to the file from which to load the model.
        """
        self.model.load_state_dict(torch.load(filepath))
