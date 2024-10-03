from torch import nn
from torch.optim import Adam

from macofl.dataset.cifar import CIFAR8DataloaderGenerator
from macofl.datatypes import ModelManager
from macofl.nn.model.cifar import CIFAR8MLP


def build_neural_network() -> ModelManager:
    cifar8_generator = CIFAR8DataloaderGenerator()
    dataloaders = cifar8_generator.get_dataloaders()
    model = CIFAR8MLP()
    return ModelManager(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.001, weight_decay=1e-5),
        batch_size=64,
        training_epochs=10,
        dataloaders=dataloaders,
    )


def test_neural_network():
    model = build_neural_network()
    training_metrics = model.train(epochs=10)
    validation_metrics = model.inference()
    test_metrics = model.test_inference()
    print(f"Training metrics: {training_metrics.accuracy} - {training_metrics.loss}")
    print(
        f"Validation metrics: {validation_metrics.accuracy} - {validation_metrics.loss}"
    )
    print(f"Test metrics: {test_metrics.accuracy} - {test_metrics.loss}")
