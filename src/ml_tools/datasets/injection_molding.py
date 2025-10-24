"""Module for handling the Injection Molding Dataset."""

import pandas as pd

from ml_tools.datasets.base import DatasetBase
from ml_tools.utils import get_data_folder


class InjectionMoldingDataset(DatasetBase):
    """Class to handle the Injection Molding Dataset."""

    def __init__(self):  # noqa: D107
        base = "InjectionMolding_{}.csv"
        data_folder = get_data_folder()
        self._train_file = data_folder / base.format("Train")
        self._test_file = data_folder / base.format("Test")
        self.raw_train_data: pd.DataFrame | None = None
        self.raw_test_data: pd.DataFrame | None = None
        self.predictor_names: list[str] | None = None
        self.response_name: str | None = None

    def _load_file(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.reset_index()
        return df

    def _load(self) -> None:
        self.raw_train_data = self._load_file(self._train_file)
        self.raw_test_data = self._load_file(self._test_file)
        self.predictor_names = self.raw_train_data.columns[0:-1].tolist()
        self.response_name = self.raw_train_data.columns[-1]

    def _split_data_labels(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = data[self.predictor_names]
        y = data[self.response_name]
        return x, y

    def _lazy_load(self) -> None:
        if self.raw_train_data is None or self.raw_test_data is None:
            self._load()

    @property
    def train_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns the training data as (X_train, y_train)."""
        self._lazy_load()
        return self._split_data_labels(self.raw_train_data)

    @property
    def test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns the testing data as (X_test, y_test)."""
        self._lazy_load()
        return self._split_data_labels(self.raw_test_data)


if __name__ == "__main__":
    dataset = InjectionMoldingDataset()
    x_train, y_train = dataset.train_data
    x_test, y_test = dataset.test_data
    print("Train features shape:", x_train.shape)
    print("Train labels shape:", y_train.shape)
    print("Test features shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)

    train_data = dataset.raw_train_data
    test_data = dataset.raw_test_data
    assert not train_data.isna().values.any()
    assert not test_data.isna().values.any()
