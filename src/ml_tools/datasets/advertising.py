"""Module for handling the Advertising Dataset."""

from pathlib import Path

import pandas as pd

from ml_tools.datasets.base import MonoDataset


class AdvertisingDataset(MonoDataset):
    """Advertising campaign dataset (ISLR, 200 observations).

    Predictors: daily `TV`, `radio` and `newspaper` advertising budgets.
    Response: `sales`, the regional product sales in thousands of dollars.
    """

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
