from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Optional

from torch.utils.data import DataLoader, Subset

from ..datatypes.loaders import DataLoaders


class DataloaderGeneratorInterface(object, metaclass=ABCMeta):

    @abstractmethod
    def get_dataloaders(
        self,
        iid: bool = True,
        preset: Optional[str] = None,
        custom_indices: Optional[Sequence[int | str]] = None,
    ) -> DataLoaders:
        raise NotImplementedError

    def _build_dataloaders(
        self, batch_size: int, train: Subset, validation: Subset, test: Subset
    ) -> DataLoaders:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
        return DataLoaders(
            train_data=train_loader,
            validation_data=validation_loader,
            test_data=test_loader,
        )
