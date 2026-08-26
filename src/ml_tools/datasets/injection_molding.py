"""Module for handling the Injection Molding Dataset."""

from typing import override

from ml_tools.datasets.base import MonoDataset
from ml_tools.utils import get_data_folder


class InjectionMoldingDataset(MonoDataset):
    """Injection molding process dataset (232 observations, fixed split).

    Predictors: 8 selected process variables and principal component scores
    (powder activity, injection position/volume, pressures, hop temperature,
    clamp force/position, oil temperature).
    Response: `mass`, the part mass in grams.
    """

    file_name: str = "InjectionMolding_{}.csv"

    @override
    def __init__(self):
        super().__init__()
        data_folder = get_data_folder()
        self._train_file = data_folder / self.file_name.format("Train")
        self._test_file = data_folder / self.file_name.format("Test")

    def _load(self) -> None:
        self.raw_train_data = self._load_file(self._train_file)
        self.raw_test_data = self._load_file(self._test_file)
        raw = self.raw_train_data
        if raw is None:
            raise RuntimeError("Training data was not loaded.")
        self.predictor_names = [str(c) for c in raw.columns[:-1]]
        self.response_name = str(raw.columns[-1])

    @property
    @override
    def is_loaded(self) -> bool:
        return self.raw_train_data is not None and self.raw_test_data is not None


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
    if train_data is not None and test_data is not None:
        assert not train_data.isna().values.any()
        assert not test_data.isna().values.any()
