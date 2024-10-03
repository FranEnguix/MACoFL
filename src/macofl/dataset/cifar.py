from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import Compose

from macofl.datatypes.loaders import DataLoaders

from .dataloader_generator import DataloaderGeneratorInterface


class CIFARN(datasets.CIFAR100):
    def __init__(
        self,
        root,
        selected_classes_names: list[str],
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        if len(selected_classes_names) == 0:
            raise ValueError("selected_classes_names must have content.")

        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.targets: list[int]
        self.class_to_idx: dict[str, int]

        selected_classes = {
            k: v for (k, v) in self.class_to_idx.items() if k in selected_classes_names
        }  # example: {'bicycle': 8, 'dolphin': 30, 'motorcycle': 48, 'ray': 67, 'shark': 73, 'tank': 85, 'tractor': 89, 'trout': 91}
        self.class_to_idx = {
            k: selected_classes_names.index(k)
            for (k, v) in self.class_to_idx.items()
            if k in selected_classes_names
        }  # example: {'bicycle': 4, 'dolphin': 3, 'motorcycle': 5, 'ray': 0, 'shark': 2, 'tank': 6, 'tractor': 7, 'trout': 1}
        self.original_class_mapping = {
            selected_classes[k]: self.class_to_idx[k] for k in selected_classes.keys()
        }  # example: {8: 4, 30: 3, 48: 5 ...}

        # Filter and remap classes
        mask = [target in self.original_class_mapping for target in self.targets]
        self.data: np.ndarray = self.data[mask]  # ndarray[N(num-samples), 32, 32, 3]
        self.targets = [
            self.original_class_mapping[target]
            for target in self.targets
            if target in self.original_class_mapping
        ]


class CIFAR8(CIFARN):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.selected_classes_names = [
            "ray",
            "trout",
            "shark",
            "dolphin",
            "bicycle",
            "motorcycle",
            "tank",
            "tractor",
        ]

        super().__init__(
            root,
            selected_classes_names=self.selected_classes_names,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def get_subset(self, labels: Sequence[int | str]) -> Subset:
        """
        Returns a torch.utils.data.Subset containing only the filtered
        data that match the labels passed by the argument.

        Args:
            labels (list[str] | list[int]): list of label names or label IDs.

        Returns:
            Subset: Torch Subset containing the filtered data.
        """
        idx_labels: list[int] = [
            self.class_to_idx[lbl] if isinstance(lbl, str) else lbl for lbl in labels
        ]

        filtered_indices_by_label = [
            idx for idx, lbl in enumerate(self.targets) if lbl in idx_labels
        ]
        return Subset(self, filtered_indices_by_label)

    @staticmethod
    def get_labels_of_superclasses() -> dict[str, list[int]]:
        animal_indices = [0, 1, 2, 3]
        vehicle_indices = [4, 5, 6, 7]
        return {"animals": animal_indices, "vehicles": vehicle_indices}


class CIFAR8DataloaderGenerator(DataloaderGeneratorInterface):
    def __init__(
        self,
        data_dir: str | Path = "premiofl_datasets/cifar8",
        batch_size: int = 8,
        train_size: float = 0.8,
        transform: Optional[Compose] = None,
    ) -> None:
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.batch_size = batch_size
        self.train_size = train_size
        self.transform = (
            Compose([transforms.ToTensor()]) if transform is None else transform
        )
        self.dataloaders: Optional[DataLoaders] = None

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.data_dir.resolve()

    def __get_cifar8_datasets(self) -> tuple[CIFAR8, CIFAR8]:
        cifar8_train = CIFAR8(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=True,
        )
        cifar8_test = CIFAR8(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=True,
        )
        return cifar8_train, cifar8_test

    def _build_dataloaders_iid(self) -> DataLoaders:
        cifar8_train, cifar8_test = self.__get_cifar8_datasets()
        train_size = int(self.train_size * len(cifar8_train))
        validation_size = len(cifar8_train) - train_size
        train_set: Subset
        validation_set: Subset
        train_set, validation_set = random_split(
            cifar8_train, [train_size, validation_size]
        )
        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=cifar8_test,
        )

    def _build_dataloaders_preset(self, preset: str) -> DataLoaders:
        label_map = CIFAR8.get_labels_of_superclasses()
        cifar8_train, cifar8_test = self.__get_cifar8_datasets()
        train_subset = cifar8_train.get_subset(labels=label_map[preset])
        train_size = int(self.train_size * len(train_subset))
        validation_size = len(train_subset) - train_size
        train_set, validation_set = random_split(
            train_subset, [train_size, validation_size]
        )
        test_set = cifar8_test.get_subset(labels=label_map[preset])
        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=test_set,
        )

    def _build_custom_dataloaders(self, labels: Sequence[int | str]) -> DataLoaders:
        cifar8_train, cifar8_test = self.__get_cifar8_datasets()
        train_subset = cifar8_train.get_subset(labels=labels)
        train_size = int(self.train_size * len(train_subset))
        validation_size = len(train_subset) - train_size
        train_set, validation_set = random_split(
            train_subset, [train_size, validation_size]
        )
        test_set = cifar8_test.get_subset(labels=labels)
        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=test_set,
        )

    def get_dataloaders(
        self,
        iid: bool = True,
        preset: Optional[str] = None,
        custom_indices: Optional[Sequence[int | str]] = None,
    ) -> DataLoaders:
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        if iid:
            return self._build_dataloaders_iid()
        if preset in ["animals", "vehicles"]:
            return self._build_dataloaders_preset(preset=preset)
        if custom_indices:
            return self._build_custom_dataloaders(labels=custom_indices)
        raise ValueError()
