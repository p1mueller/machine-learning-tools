"""Base class for datasets."""

import abc
from pathlib import Path
from typing import Self, override

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_tools.utils import get_data_folder

FeatureTarget = tuple[pd.DataFrame, pd.Series]


class DatasetBase(abc.ABC):
    """Abstract base class for datasets."""

    @property
    @abc.abstractmethod
    def train_data(self) -> FeatureTarget:
        """Return training data as a tuple of (features, targets)."""

    @property
    @abc.abstractmethod
    def test_data(self) -> FeatureTarget:
        """Return test data as a tuple of (features, targets)."""


class LazyDataset(DatasetBase):
    """Abstract class that loads a dataset lazily."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Flag if data was already loaded."""

    @abc.abstractmethod
    def _load(self) -> None:
        """Load the underlying data."""

    def load(self) -> Self:
        """Load the dataset."""
        if not self.is_loaded:
            self._load()
        return self


class MonoDataset(LazyDataset):
    """Dataset that consist only of csv file(s)."""

    file_name: str

    def __init__(self, split: float = 0.7) -> None:
        """Initialize.

        Args:
            split: Fraction of data used for training
        """
        self._file = get_data_folder() / self.file_name
        self._split = split
        self.raw_train_data: pd.DataFrame | None = None
        self.raw_test_data: pd.DataFrame | None = None
        self.predictor_names: list[str] | None = None
        self.response_name: str | None = None

    @property
    @override
    def is_loaded(self) -> bool:
        return self.raw_train_data is not None and self.raw_test_data is not None

    def _load_file(self, file: str | Path) -> pd.DataFrame:
        return pd.read_csv(file)

    @override
    def _load(self) -> None:
        df = self._load_file(self._file)
        self.raw_train_data, self.raw_test_data = train_test_split(
            df, train_size=self._split, shuffle=False
        )

    def _split_data_and_labels(self, data: pd.DataFrame | None) -> FeatureTarget:
        if (data is None) or (self.predictor_names is None) or (self.response_name is None):
            raise RuntimeError("Dataset has not been loaded.")
        x = data[self.predictor_names]
        y = data[self.response_name]
        return x, y

    @property
    @override
    def train_data(self) -> FeatureTarget:
        self.load()
        return self._split_data_and_labels(self.raw_train_data)

    @property
    @override
    def test_data(self) -> FeatureTarget:
        self.load()
        return self._split_data_and_labels(self.raw_test_data)
