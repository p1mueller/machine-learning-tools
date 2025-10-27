"""Test for datasets module."""

import numpy as np
import pytest

import ml_tools.datasets as ds

np.random.seed(42)


def _check_dataset(
    dataset: ds.AutoDataset | ds.AdvertisingDataset | ds.InjectionMoldingDataset,
    predictors: list[str],
    response: str,
    train_samples: int,
    test_samples: int,
    check_unique: bool = True,
) -> None:
    assert dataset.raw_test_data is None
    assert dataset.raw_test_data is None
    assert dataset.predictor_names is None
    assert dataset.response_name is None

    x_train, y_train = dataset.train_data
    assert dataset.predictor_names == predictors
    assert dataset.response_name == response
    assert dataset.raw_train_data is not None
    assert x_train.shape == (train_samples, len(predictors))
    assert y_train.shape == x_train.shape[:1]

    x_test, y_test = dataset.test_data
    assert dataset.predictor_names == predictors
    assert dataset.response_name == response
    assert dataset.raw_test_data is not None
    assert x_test.shape == (test_samples, len(predictors))
    assert y_test.shape == x_test.shape[:1]

    if check_unique:
        dist = np.linalg.norm(np.array(x_train)[:, None] - np.array(x_test)[None], axis=2)
        assert dist.min() > 1e-12


def test_injection() -> None:
    """Check injection molding dataset."""
    train_samples = 150
    test_samples = 82
    predictors = [
        "PowTotAct_Min",
        "Inj1PosVolAct_Var",
        "Inj1PrsAct_meanOfInjPhase",
        "Inj1HopTmpAct_1stPCscore",
        "Inj1HtgEd3Act_1stPCscore",
        "ClpFceAct_1stPCscore",
        "ClpPosAct_1stPCscore",
        "OilTmp1Act_1stPCscore",
    ]
    response = "mass"
    dataset = ds.InjectionMoldingDataset()
    _check_dataset(dataset, predictors, response, train_samples, test_samples)


def test_advertising() -> None:
    """Check advertising dataset."""
    n_samples = 200
    predictors = ["TV", "radio", "newspaper"]
    response = "sales"
    split = 0.8
    dataset = ds.AdvertisingDataset(split=split)
    train_samples = int(split * n_samples)
    test_samples = n_samples - train_samples

    _check_dataset(dataset, predictors, response, train_samples, test_samples)


def test_auto() -> None:
    """Check auto dataset."""
    split = 0.8
    train_samples = 315
    test_samples = 80
    response = "mpg"
    columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "year",
        "origin",
        "name",
    ]
    predictors = columns.copy()
    predictors.remove(response)
    dataset = ds.AutoDataset(split=split)
    _check_dataset(dataset, predictors, response, train_samples, test_samples, check_unique=False)


if __name__ == "__main__":
    pytest.main([__file__])
