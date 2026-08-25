"""Module for handling the Advertising Dataset."""

from pathlib import Path

import pandas as pd

from ml_tools.datasets.base import MonoDataset


class AdvertisingDataset(MonoDataset):
    """Class to handle the Advertising Dataset."""

    file_name: str = "Advertising.csv"

    def _load_file(self, file: str | Path) -> pd.DataFrame:
        df = super()._load_file(file)
        df = df.drop(columns=df.columns[0])
        return df

    def _load(self) -> None:
        super()._load()
        raw = self.raw_train_data
        if raw is None:
            raise RuntimeError("Training data was not loaded by the base class.")
        columns = [str(c) for c in raw.columns]
        self.predictor_names = columns[:-1]
        self.response_name = columns[-1]


if __name__ == "__main__":
    dataset = AdvertisingDataset()
    x_train, y_train = dataset.train_data
    x_test, y_test = dataset.test_data

    train = dataset.raw_train_data
    if train is not None:
        print("Advertising Dataset")
        print(train.head())

    print("Train features shape:", x_train.shape)
    print("Train labels shape:", y_train.shape)
    print("Test features shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)

    train_data = dataset.raw_train_data
    test_data = dataset.raw_test_data

    print(x_train.columns.to_list(), dataset.predictor_names)
