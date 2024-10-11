import unittest

import pytest

from macofl.dataset import cifar, mnist
from macofl.dataset.dataloader_generator import BaseDataLoaderGenerator


class TestDataLoader(unittest.TestCase):

    @pytest.mark.skip(reason="Helper method, not a test")
    def test_iid_non_iid(
        self, generator: BaseDataLoaderGenerator, num_clients: int = 2
    ):
        iid = generator.get_dataloaders(iid=True, client_index=1)
        train_size = len(iid.train)

        diritchet_size = 0
        for client_index in range(num_clients):
            non_iids = generator.get_dataloaders(
                iid=False,
                client_index=client_index,
                num_clients=num_clients,
                dirichlet_alpha=0.1,
            )
            diritchet_size += len(non_iids.train)

        assert (
            diritchet_size + num_clients >= train_size >= diritchet_size - num_clients
        ), "The number of batchs of IID and non-IID differs more than expected."

    def test_mnist_dataset(self) -> None:
        self.test_iid_non_iid(mnist.MnistDataLoaderGenerator())

    def test_cifar10_dataset(self) -> None:
        self.test_iid_non_iid(cifar.Cifar10DataLoaderGenerator())

    def test_cifar100_dataset(self) -> None:
        self.test_iid_non_iid(cifar.Cifar100DataLoaderGenerator())
