"""Auto Dataset."""

from pathlib import Path

import pandas as pd

from ml_tools.datasets.base import MonoDataset


class AutoDataset(MonoDataset):
    """Auto-mpg fleet dataset (UCI/statlearning, 392 observations).

    Predictors: `cylinders`, `displacement`, `horsepower`, `weight`,
    `acceleration`, `year` and `origin`.
    Response: `mpg`, the miles per gallon fuel consumption.
    """

    file_name = "Auto.csv"

    def __init__(self, response_name: str | None = "mpg", split: float = 0.7) -> None:
        """Initialize.

        Args:
            response_name (str | None): Name of the response variable.
                If None, 'mpg' is used.
            split: Fraction of data used for training
        """
        super().__init__(split=split)
        self._response_name = response_name

    def _load_file(self, file: str | Path) -> pd.DataFrame:
        df = super()._load_file(file)
        df = df.replace("?", float("nan"))
        numeric_columns = [col for col in df.columns if col != "name"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        return df

    def _load(self) -> None:
        super()._load()
        raw = self.raw_train_data
        if raw is None:
            raise RuntimeError("Training data was not loaded by the base class.")
        raw = raw.dropna()
        self.raw_train_data = raw
        raw_test = self.raw_test_data
        if raw_test is not None:
            self.raw_test_data = raw_test.dropna()
        numeric_columns = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        response = self._response_name
        if response is None:
            response = numeric_columns[-1]
        if response not in numeric_columns:
            raise ValueError(f"Response column '{response}' is not numeric.")
        self.response_name = response
        self.predictor_names = [c for c in numeric_columns if c != response]
