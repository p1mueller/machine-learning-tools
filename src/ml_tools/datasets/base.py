"""Base class for datasets."""

import abc
from typing import Self, override

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_tools.utils import get_data_folder

Data = pd.DataFrame | np.ndarray


class DatasetBase:
    """Abstract base class for datasets."""

    @property
    @abc.abstractmethod
    def train_data(self) -> tuple[Data, Data]:
        """Return training data as a tuple of (features, targets)."""
        pass

    @property
    @abc.abstractmethod
    def test_data(self) -> tuple[Data, Data]:
        """Return test data as a tuple of (features, targets)."""
        pass


class LazyDataset(DatasetBase):
    """Abstract class that loads a dataset lazily."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Flag if data was already loaded."""
        pass

    @abc.abstractmethod
    def _load(self) -> None:
        pass

    def load(self) -> Self:
        """Load the dataset."""
        if not self.is_loaded:
            self._load()


class MonoDataset(LazyDataset):
    """Dataset that consist only of one csv file."""

    file_name: str

    def __init__(self, split: float = 0.7):
        """Initialize.

        Args:
            split: Fraction of data used for training
        """
        self._file = get_data_folder() / self.file_name
        self._split = split
        self.raw_train_data: Data | None = None
        self.raw_test_data: Data | None = None
        self.predictor_names: list[str] | None = None
        self.response_name: str | None = None

    @property
    @override
    def is_loaded(self) -> bool:
        return self.raw_train_data is not None and self.raw_test_data is not None

    def _load_file(self, file: str) -> pd.DataFrame:
        df = pd.read_csv(file)
        return df

    def _load(self):
        df = self._load_file(self._file)
        self.raw_train_data, self.raw_test_data = train_test_split(
            df, train_size=self._split, shuffle=False
        )

    def _split_data_and_labels(self, data: Data) -> tuple[Data, Data]:
        x = data[self.predictor_names]
        y = data[self.response_name]
        return x, y

    @property
    @override
    def train_data(self) -> tuple[Data, Data]:
        self.load()
        return self._split_data_and_labels(self.raw_train_data)

    @property
    @override
    def test_data(self) -> tuple[Data, Data]:
        self.load()
        return self._split_data_and_labels(self.raw_test_data)
