from dataclasses import dataclass
from datetime import datetime, timedelta
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
    start_time_z: Optional[datetime] = None
    end_time_z: Optional[datetime] = None

    def time_elapsed(self) -> timedelta:
        if not self.end_time_z or not self.start_time_z:
            raise ValueError("datetime is None in time_elapsed ModelMetric class.")
        return self.end_time_z - self.start_time_z
