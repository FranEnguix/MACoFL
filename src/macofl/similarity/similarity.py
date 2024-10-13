from abc import ABCMeta, abstractmethod
from typing import OrderedDict

from torch import Tensor

from .similarity_vector import SimilarityVector


class SimilarityFunction(object, metaclass=ABCMeta):

    @abstractmethod
    @staticmethod
    def get_similarity_vector(
        layers1: OrderedDict[str, Tensor],
        layers2: OrderedDict[str, Tensor],
    ) -> SimilarityVector:
        raise NotImplementedError
