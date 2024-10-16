from torch import nn
from torch.optim import Adam

from ..dataset.cifar import Cifar10DataLoaderGenerator
from ..datatypes.data import DatasetSettings
from ..datatypes.models import ModelManager
from ..utils.random import RandomUtils
from .model.cnn import CNN5
from .model.mlp import CifarMlp


class ModelManagerFactory:

    @staticmethod
    def get_cifar10_mlp(settings: DatasetSettings) -> ModelManager:
        cifar10_generator = Cifar10DataLoaderGenerator()
        dataloaders = cifar10_generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = CifarMlp(input_dim=(3, 32, 32), out_classes=10)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0,
                amsgrad=False,
            ),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=settings.seed,
        )

    @staticmethod
    def get_cifar10_cnn5(settings: DatasetSettings) -> ModelManager:
        cifar10_generator = Cifar10DataLoaderGenerator()
        dataloaders = cifar10_generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = CNN5(input_dim=(3, 32, 32), out_classes=10)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0,
                amsgrad=False,
            ),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=settings.seed,
        )

    @staticmethod
    def get_cifar100_cnn5(settings: DatasetSettings) -> ModelManager:
        cifar10_generator = Cifar10DataLoaderGenerator()
        dataloaders = cifar10_generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = CNN5(input_dim=(3, 32, 32), out_classes=100)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0,
                amsgrad=False,
            ),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=settings.seed,
        )

    # @staticmethod
    # def get_mnist_cnn5(settings: DatasetSettings) -> ModelManager:
    #     cifar10_generator = Mn()
    #     dataloaders = cifar10_generator.get_dataloaders(settings=settings)
    #     RandomUtils.set_randomness(seed=settings.seed)
    #     model = CNN5(input_dim=(1, 28, 28), out_classes=10)
    #     return ModelManager(
    #         model=model,
    #         criterion=nn.CrossEntropyLoss(),
    #         optimizer=Adam(
    #             model.parameters(),
    #             lr=1e-3,
    #             betas=(0.9, 0.999),
    #             eps=1e-7,
    #             weight_decay=0,
    #             amsgrad=False,
    #         ),
    #         batch_size=64,
    #         training_epochs=1,
    #         dataloaders=dataloaders,
    #         seed=settings.seed,
    #     )
