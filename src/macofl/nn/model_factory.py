from typing import Optional

from torch import nn
from torch.optim import Adam

from ..dataset.cifar import Cifar10DataLoaderGenerator
from ..datatypes.models import ModelManager
from ..utils.random import RandomUtils
from .model.cifar import CifarMlp


class ModelManagerFactory:

    @staticmethod
    def get_cifar10(
        iid: bool,
        client_index: int = 0,
        num_clients: int = 10,
        dirichlet_alpha: float = 0.1,
        seed: Optional[int] = 42,
    ) -> ModelManager:
        cifar10_generator = Cifar10DataLoaderGenerator()
        dataloaders = cifar10_generator.get_dataloaders(
            iid=iid,
            client_index=client_index,
            num_clients=num_clients,
            dirichlet_alpha=dirichlet_alpha,
            seed=seed,
        )
        RandomUtils.set_randomness(seed=seed)
        model = CifarMlp(classes=10)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(model.parameters(), lr=0.001, weight_decay=1e-5),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=seed,
        )
