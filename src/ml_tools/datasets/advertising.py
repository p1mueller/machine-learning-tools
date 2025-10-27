"""Module for handling the Advertising Dataset."""

from ml_tools.datasets.base import MonoDataset


class AdvertisingDataset(MonoDataset):
    """Class to handle the Injection Molding Dataset."""

    file_name: str = "Advertising.csv"

    def _load_file(self, file):
        df = super()._load_file(file)
        df = df.drop(df.columns[0], axis=1)
        return df

    def _load(self) -> None:
        super()._load()
        self.predictor_names = self.raw_train_data.columns[:-1].tolist()
        self.response_name = self.raw_train_data.columns[-1]


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

    print(x_train.columns.to_list(), dataset.predictor_names)
