"""Auto Dataset."""

import pandas as pd

from ml_tools.datasets.base import MonoDataset


class AutoDataset(MonoDataset):
    """Class to handle the Injection Molding Dataset."""

    file_name = "Auto.csv"

    def __init__(self, response_name: str | None = "mpg", split: int = 0.7):
        """Initialize.

        Args:
            response_name (str | None): Name of the response variable.
                If None, 'mpg' is used.
            split: Fraction of data used for training
        """
        super().__init__(split=split)
        self._response_name = response_name

    def _load_file(self, file_path: str):
        df = super()._load_file(file_path)
        df = df.replace("?", float("nan"))
        numeric_columns = df.columns.tolist()
        numeric_columns.remove("name")
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        return df

    def _load(self) -> None:
        super()._load()
        self.raw_train_data.dropna(inplace=True)
        if self._response_name is None:
            self.response_name = self.raw_train_data.columns[-1]
        else:
            self.response_name = self._response_name
        predictors = self.raw_train_data.columns.tolist()
        predictors.remove(self.response_name)
        self.predictor_names = predictors


if __name__ == "__main__":
    dataset = AutoDataset()
    x_train, y_train = dataset.train_data
    x_test, y_test = dataset.test_data

    print("Auto Dataset")
    print(dataset.raw_train_data.head())

    print("Train features shape:", x_train.shape)
    print("Train labels shape:", y_train.shape)
    print("Test features shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)

    train_data = dataset.raw_train_data
    test_data = dataset.raw_test_data
    print(dataset.predictor_names)
