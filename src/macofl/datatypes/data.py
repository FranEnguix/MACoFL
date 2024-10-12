from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


@dataclass
class DatasetSettings(object, metaclass=ABCMeta):
    iid: bool
    seed: Optional[int]


class IidDatasetSettings(DatasetSettings):
    def __init__(
        self,
        seed: Optional[int],
        samples_in_percent: Optional[bool] = None,
        train_samples: Optional[float | int] = None,
        test_samples: Optional[float | int] = None,
    ) -> None:
        super().__init__(iid=True, seed=seed)
        self.samples_in_percent = samples_in_percent
        self._train_samples = train_samples
        self._test_samples = test_samples

        if not self.are_all_samples_selected():
            is_percent = isinstance(self._train_samples, float) and isinstance(
                self._test_samples, float
            )
            is_number_of_samples = isinstance(self._train_samples, int) and isinstance(
                self._test_samples, int
            )
            if not (
                (is_percent and samples_in_percent)
                or (is_number_of_samples and not samples_in_percent)
            ):
                raise ValueError(
                    "All samples must be either percentage or number of samples."
                )

    @property
    def train_samples(self) -> float | int:
        if self.are_all_samples_selected() or self._train_samples is None:
            raise RuntimeError(
                "Trying to get a subset of train samples when all are selected."
            )
        return self._train_samples

    @property
    def test_samples(self) -> float | int:
        if self.are_all_samples_selected() or self._test_samples is None:
            raise RuntimeError(
                "Trying to get a subset of test samples when all are selected."
            )
        return self._test_samples

    def are_all_samples_selected(self) -> bool:
        return self.samples_in_percent is None

    def is_percent_of_samples(self) -> bool:
        return self.samples_in_percent is not None and self.samples_in_percent


class NonIidDatasetSettings(DatasetSettings, metaclass=ABCMeta):

    def __init__(
        self, seed: Optional[int], num_clients: int, client_index: int
    ) -> None:
        super().__init__(iid=False, seed=seed)
        self.num_clients = num_clients
        self.client_index = client_index


class NonIidNonOverlappingClassesDatasetSettings(NonIidDatasetSettings):
    def __init__(
        self,
        seed: Optional[int],
        num_clients: int,
        client_index: int,
        classes_per_client: int,
    ) -> None:
        super().__init__(seed=seed, num_clients=num_clients, client_index=client_index)
        self.classes_per_client = classes_per_client


class NonIidDirichletDatasetSettings(NonIidDatasetSettings):
    def __init__(
        self,
        seed: Optional[int],
        num_clients: int,
        client_index: int,
        dirichlet_alpha: float = 0.1,
    ) -> None:
        super().__init__(seed=seed, num_clients=num_clients, client_index=client_index)
        self.dirichlet_alpha = dirichlet_alpha
