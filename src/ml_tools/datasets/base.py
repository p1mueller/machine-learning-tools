import abc

import numpy as np
import pandas as pd

Data = pd.DataFrame | np.ndarray


class DatasetBase:
    """Abstract base class for datasets."""

    @property
    @abc.abstractmethod
    def train_data(self) -> tuple[Data, Data]:
        pass

    @property
    @abc.abstractmethod
    def test_data(self) -> tuple[Data, Data]:
        pass
