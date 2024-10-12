from torch import nn
from torch.optim import Adam

from ..dataset.cifar import Cifar10DataLoaderGenerator
from ..datatypes.data import DatasetSettings
from ..datatypes.models import ModelManager
from ..utils.random import RandomUtils
from .model.cifar import CifarMlp


class ModelManagerFactory:

    @staticmethod
    def get_cifar10(settings: DatasetSettings) -> ModelManager:
        cifar10_generator = Cifar10DataLoaderGenerator()
        dataloaders = cifar10_generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = CifarMlp(classes=10)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(model.parameters(), lr=0.001, weight_decay=1e-5),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=settings.seed,
        )
