import torch
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
        training_epochs=1,
        dataloaders=dataloaders,
    )


def test_neural_network():
    model = build_neural_network()
    training_metrics = model.train()
    validation_metrics = model.inference()
    test_metrics = model.test_inference()
    print(f"Training metrics: {training_metrics.accuracy} - {training_metrics.loss}")
    print(
        f"Validation metrics: {validation_metrics.accuracy} - {validation_metrics.loss}"
    )
    print(f"Test metrics: {test_metrics.accuracy} - {test_metrics.loss}")


def test_model_to_base64():
    model = build_neural_network()
    model_str = ModelManager.export_weights_and_biases(model.model.state_dict())
    model_reconstruct = ModelManager.import_weights_and_biases(model_str)
    for key in model.initial_state:
        assert key in model_reconstruct, f"Key '{key}' not in reconstruct."
        assert torch.allclose(
            model.model.state_dict()[key], model_reconstruct[key]
        ), f"Reconstructed '{key}' tensor does not match the model"
        assert torch.allclose(
            model.initial_state[key], model_reconstruct[key]
        ), f"Reconstructed '{key}' tensor does not match the initial model"
