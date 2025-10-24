"""Auto Dataset."""

import pandas as pd

from ml_tools.datasets.injection_molding import InjectionMoldingDataset
from ml_tools.utils import get_data_folder


class AutoDataset(InjectionMoldingDataset):
    """Class to handle the Injection Molding Dataset."""

    def __init__(self, response_name: str | None = "mpg"):
        """Initialize.

        Args:
            response_name (str | None): Name of the response variable.
                If None, 'mpg' is used.
        """
        super().__init__()
        base = "Auto.csv"
        data_folder = get_data_folder()
        self._response_name = response_name
        self._train_file = data_folder / base
        self._test_file = self._train_file  # Using the same file for test for demonstration

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
    assert not train_data.isna().values.any()
    assert not test_data.isna().values.any()
