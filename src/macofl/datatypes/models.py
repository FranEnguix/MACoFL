import codecs
import copy
import pickle
from datetime import datetime, timezone
from typing import Callable, Optional, OrderedDict

import torch
from aioxmpp import JID
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..datatypes.metrics import ModelMetrics

# from ..utils.random import RandomUtils
from .data import DataLoaders


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
        seed: Optional[int] = 42,
        deterministic: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.dataloaders = dataloaders
        self.seed = seed
        self.deterministic = deterministic
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        # TODO when the below TODO is completed: RandomUtils.set_randomness(seed=self.seed)
        # TODO Ask for a model generator and generate the model here: self.model = generator.get_model(parameters)
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

    def train(
        self,
        epochs: Optional[int] = None,
        train_logger: Optional[Callable[[int, ModelMetrics, JID, int], None]] = None,
        agent_jid: Optional[JID] = None,
        algorithm_iteration: Optional[int] = None,
    ) -> list[ModelMetrics]:
        """
        Updates the model by training on the training dataset.
        """
        self.pretrain_state = copy.deepcopy(self.model.state_dict())
        self.__training = True
        if epochs is None:
            epochs = self.training_epochs

        metrics: list[ModelMetrics] = []

        # Training loop
        try:
            for epoch in range(epochs):
                self.model.train()
                total_loss: float = 0.0
                correct: int = 0
                total_samples: int = 0

                images: Tensor
                labels: Tensor
                outputs: Tensor
                loss: Tensor
                predicted: Tensor

                init_time_z = datetime.now(tz=timezone.utc)
                for images, labels in self.dataloaders.train:
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

                epoch_metric: ModelMetrics = ModelMetrics(
                    accuracy=(correct / total_samples),
                    loss=(total_loss / len(self.dataloaders.train)),
                    start_time_z=init_time_z,
                    end_time_z=datetime.now(tz=timezone.utc),
                )
                metrics.append(epoch_metric)
                if (
                    train_logger is not None
                    and agent_jid is not None
                    and algorithm_iteration is not None
                ):
                    train_logger(
                        epoch + 1, epoch_metric, agent_jid, algorithm_iteration
                    )
            return metrics

        finally:
            self.__training = False

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

        init_time_z = datetime.now(tz=timezone.utc)
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

        end_time_z: datetime = datetime.now(tz=timezone.utc)
        accuracy: float = correct / total
        resulting_loss: float = total_loss / len(dataloader)
        metrics: ModelMetrics = ModelMetrics(
            accuracy=accuracy,
            loss=resulting_loss,
            start_time_z=init_time_z,
            end_time_z=end_time_z,
        )
        return metrics

    def inference(self) -> ModelMetrics:
        """
        Returns the TRAIN inference metrics using validation data.
        """
        return self._inference(dataloader=self.dataloaders.validation)

    def test_inference(self) -> ModelMetrics:
        """
        Returns the TEST inference metrics.
        """
        return self._inference(dataloader=self.dataloaders.test)

    def get_layers(self, layers: list[str]) -> OrderedDict[str, Tensor]:
        part_of_model: OrderedDict[str, Tensor] = OrderedDict()
        for layer in layers:
            part_of_model[layer] = self.model.state_dict()[layer]
        return part_of_model

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
