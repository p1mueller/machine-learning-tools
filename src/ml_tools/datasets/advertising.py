from ml_tools.datasets.injection_molding import InjectionMoldingDataset
from ml_tools.utils import get_data_folder


class AdvertisingDataset(InjectionMoldingDataset):
    """Class to handle the Injection Molding Dataset."""

    def __init__(self):
        super().__init__()
        base = "Advertising.csv"
        data_folder = get_data_folder()
        self._train_file = data_folder / base
        self._test_file = self._train_file  # Using the same file for test for demonstration

    def _load(self) -> None:
        super()._load()
        self.predictor_names = self.raw_train_data.columns[1:-1].tolist()


if __name__ == "__main__":
    dataset = AdvertisingDataset()
    x_train, y_train = dataset.train_data
    x_test, y_test = dataset.test_data

    print("Advertising Dataset")
    print(dataset.raw_train_data.head())

    print("Train features shape:", x_train.shape)
    print("Train labels shape:", y_train.shape)
    print("Test features shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)

    train_data = dataset.raw_train_data
    test_data = dataset.raw_test_data
