from torch import nn
from torch.optim import Adam

from ..dataset.cifar import CIFAR8DataloaderGenerator
from ..datatypes.models import ModelManager
from .model.cifar import CIFAR8MLP


class ModelManagerFactory:

    @staticmethod
    def get_cifar8() -> ModelManager:
        cifar8_generator = CIFAR8DataloaderGenerator()
        dataloaders = cifar8_generator.get_dataloaders()
        model = CIFAR8MLP()
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(model.parameters(), lr=0.001, weight_decay=1e-5),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
        )
