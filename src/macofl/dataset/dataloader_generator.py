from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Compose

from macofl.datatypes.loaders import DataLoaders

from ..datatypes.loaders import DataLoaders
from ..utils.random import RandomUtils


class DataloaderGeneratorInterface(object, metaclass=ABCMeta):

    @abstractmethod
    def get_dataloaders(
        self,
        iid: bool,
        client_index: int,
        num_clients: int = 10,
        dirichlet_alpha: float = 0.1,
    ) -> DataLoaders:
        """Gets the data loaders for IID or Non-IID data.

        Args:
            iid (bool): If True, generates IID data loaders; otherwise, Non-IID.
            num_clients (int): Number of clients for Non-IID data.
            classes_per_client (int): Number of classes per client for Non-IID data.

        Returns:
            DataLoaders: Dataclass containing the data loaders.
        """
        raise NotImplementedError

    def _build_dataloaders(
        self, batch_size: int, train: Dataset, validation: Dataset, test: Dataset
    ) -> DataLoaders:
        """Builds data loaders for training, validation, and testing.

        Args:
            batch_size (int): Batch size for data loaders.
            train (Dataset): Training dataset.
            validation (Dataset): Validation dataset.
            test (Dataset): Test dataset.

        Returns:
            DataLoaders: Dataclass containing the data loaders.
        """
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
        return DataLoaders(
            train=train_loader,
            validation=validation_loader,
            test=test_loader,
        )


class BaseDataLoaderGenerator(DataloaderGeneratorInterface):
    """Base data loader generator class for handling IID and Non-IID data."""

    def __init__(
        self,
        dataset_cls: type,
        data_dir: str | Path,
        batch_size: int,
        train_size: float,
        transform: Optional[Compose] = None,
    ) -> None:
        """Initializes the data loader generator.

        Args:
            dataset_cls (type): The dataset class (e.g., datasets.CIFAR10).
            data_dir (str | Path): Directory where data will be stored.
            batch_size (int): Batch size for data loaders.
            train_size (float): Proportion of data to use for training.
            transform (Optional[Compose]): Transformations to apply to the data.
        """
        self.dataset_cls = dataset_cls
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.batch_size = batch_size
        self.train_size = train_size
        self.transform = transform or Compose([transforms.ToTensor()])

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.data_dir.resolve()

    def _get_datasets(self) -> Tuple[VisionDataset, VisionDataset]:
        """Loads the train and test datasets.

        Returns:
            Tuple[VisionDataset, VisionDataset]: Train and test datasets.
        """
        train_dataset = self.dataset_cls(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=True,
        )
        test_dataset = self.dataset_cls(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=True,
        )
        return train_dataset, test_dataset

    def _build_dataloaders_iid(self) -> DataLoaders:
        """Builds IID data loaders.

        Returns:
            DataLoaders: Data loaders for IID data.
        """
        train_dataset, test_dataset = self._get_datasets()
        train_size = int(self.train_size * len(train_dataset))
        validation_size = len(train_dataset) - train_size
        train_set, validation_set = random_split(
            train_dataset, [train_size, validation_size]
        )
        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=test_dataset,
        )

    def _build_dataloaders_non_iid(
        self, num_clients: int, classes_per_client: int
    ) -> DataLoaders:
        """Builds Non-IID data loaders.

        Args:
            num_clients (int): Number of clients.
            classes_per_client (int): Number of classes per client.

        Returns:
            DataLoaders: Data loaders for Non-IID data.
        """
        train_dataset, test_dataset = self._get_datasets()
        num_classes = len(train_dataset.classes)
        print(f"num classes: {num_classes}")

        # Create a mapping from class indices to data indices
        targets = np.array(train_dataset.targets)
        data_indices = [np.where(targets == i)[0] for i in range(num_classes)]
        print(f"data_indices: {data_indices[:10]}")

        # Shuffle and distribute classes to clients
        classes = list(range(num_classes))
        np.random.shuffle(classes)
        if num_clients * classes_per_client <= num_classes:
            raise RuntimeError(
                f"Not enough classes {num_classes} for the number of clients {num_clients} and {classes_per_client}."
            )

        client_classes = [
            classes[i * classes_per_client : (i + 1) * classes_per_client]
            for i in range(num_clients)
        ]

        # Collect indices for each client
        client_indices: list[int] = []
        for cls in client_classes:
            indices = np.hstack([data_indices[c] for c in cls])
            np.random.shuffle(indices)
            client_indices.extend(indices)

        # Split into training and validation sets
        all_train_indices = np.hstack(client_indices)
        train_size = int(self.train_size * len(all_train_indices))
        train_indices: list[int] = all_train_indices[:train_size].tolist()
        validation_indices: list[int] = all_train_indices[train_size:].tolist()

        train_set = Subset(train_dataset, train_indices)
        validation_set = Subset(train_dataset, validation_indices)

        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=test_dataset,
        )

    def _build_dataloaders_non_iid_dirichlet(
        self, client_index: int, num_clients: int, alpha: float
    ) -> DataLoaders:
        """Builds Non-IID data loaders using a Dirichlet distribution.

        Args:
            num_clients (int): Number of clients.
            alpha (float): Parameter controlling the level of non-IID-ness.

        Returns:
            DataLoaders: Data loaders for Non-IID data.
        """
        train_dataset, test_dataset = self._get_datasets()
        num_classes = len(train_dataset.classes)
        targets = np.array(train_dataset.targets)
        # data_indices = np.arange(len(train_dataset))

        # Generate Dirichlet distribution
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
        client_data_indices: list[list] = [[] for _ in range(num_clients)]

        for _, indices in enumerate(class_indices):
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (proportions * len(indices)).astype(int)
            proportions[-1] = len(indices) - sum(proportions[:-1])  # Adjust last client
            split_indices = np.split(indices, np.cumsum(proportions)[:-1])

            # for i, client_indices_i in enumerate(split_indices):
            #     client_data_indices[i].extend(client_indices_i)
            client_data_indices[client_index].extend(split_indices[client_index])

        # Flatten and shuffle indices
        # all_train_indices = np.hstack(client_data_indices)
        np.random.shuffle(client_data_indices[client_index])
        train_size = int(self.train_size * len(client_data_indices[client_index]))

        train_indices = client_data_indices[client_index][:train_size]
        validation_indices = client_data_indices[client_index][train_size:]

        train_set = Subset(train_dataset, train_indices)
        validation_set = Subset(train_dataset, validation_indices)

        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=test_dataset,
        )

    def get_dataloaders(
        self,
        iid: bool,
        client_index: int = 0,
        num_clients: int = 10,
        dirichlet_alpha: float = 0.1,
        seed: Optional[int] = 42,
    ) -> DataLoaders:
        RandomUtils.set_randomness(seed=seed)
        if iid:
            return self._build_dataloaders_iid()
        return self._build_dataloaders_non_iid_dirichlet(
            client_index=client_index, num_clients=num_clients, alpha=dirichlet_alpha
        )
