import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam

from ..dataset.cifar import CIFAR8DataloaderGenerator
from ..datatypes.models import ModelManager
from .model.cifar import CIFAR8MLP


class ModelManagerFactory:

    @staticmethod
    def set_randomness(seed: Optional[int] = 42) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            cudnn.deterministic = True
            cudnn.benchmark = False

    @staticmethod
    def get_cifar8(seed: Optional[int] = 42) -> ModelManager:
        ModelManagerFactory.set_randomness(seed)
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
            seed=seed,
        )
