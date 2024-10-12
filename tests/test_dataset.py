import unittest

from macofl.dataset import cifar, mnist
from macofl.dataset.dataloader_generator import BaseDataLoaderGenerator
from macofl.datatypes.data import IidDatasetSettings, NonIidDirichletDatasetSettings


class TestDataLoader(unittest.TestCase):

    def get_generators(self) -> list[BaseDataLoaderGenerator]:
        return [
            mnist.MnistDataLoaderGenerator(),
            cifar.Cifar10DataLoaderGenerator(),
            cifar.Cifar100DataLoaderGenerator(),
        ]

    # @pytest.mark.skip(reason="Helper method, not a test")
    def test_full_iid_equivalent_to_non_iid(self, num_clients: int = 2):
        iid_settings = IidDatasetSettings(seed=42)
        for generator in self.get_generators():
            iid = generator.get_dataloaders(settings=iid_settings)
            train_batches = len(iid.train)

            diritchet_batches = 0
            for client_index in range(num_clients):
                non_iid_settings = NonIidDirichletDatasetSettings(
                    seed=42, num_clients=num_clients, client_index=client_index
                )
                non_iids = generator.get_dataloaders(settings=non_iid_settings)
                diritchet_batches += len(non_iids.train)

            assert (
                diritchet_batches - num_clients
                <= train_batches
                <= diritchet_batches + num_clients
            ), "The number of batchs of IID and non-IID differs more than expected."

    def test_reduced_iid(self, new_size: float = 0.1):
        iid_reduced_settings = IidDatasetSettings(
            seed=42,
            samples_in_percent=True,
            train_samples=new_size,
            test_samples=new_size,
        )
        iid_full_settings = IidDatasetSettings(seed=42)
        for generator in self.get_generators():
            full = generator.get_dataloaders(settings=iid_full_settings)
            reduced = generator.get_dataloaders(settings=iid_reduced_settings)
            batches = [
                (len(full.train), len(reduced.train)),
                (len(full.validation), len(reduced.validation)),
                (len(full.test), len(reduced.test)),
            ]
            for batches_full, batches_reduced in batches:
                assert (
                    batches_full * (new_size + 0.02)
                    >= batches_reduced
                    >= batches_full * (new_size - 0.02)
                )
