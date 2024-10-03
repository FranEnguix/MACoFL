from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelMetrics:
    """
    Dataclass to store various performance metrics of a model evaluation.
    """

    accuracy: float
    loss: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
