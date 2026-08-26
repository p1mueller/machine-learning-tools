"""Test for datasets module."""

import numpy as np
import pandas as pd
import pytest

import ml_tools.datasets as ds

np.random.seed(42)


def _check_dataset(
    dataset: ds.AutoDataset | ds.AdvertisingDataset | ds.InjectionMoldingDataset,
    predictors: list[str],
    response: str,
    train_samples: int,
    test_samples: int,
) -> None:
    assert dataset.raw_train_data is None
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
    test_samples = 77
    response = "mpg"
    predictors = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "year",
        "origin",
    ]
    dataset = ds.AutoDataset(split=split)
    _check_dataset(dataset, predictors, response, train_samples, test_samples)


def test_auto_response_fallback() -> None:
    """A None response defaults to the last numeric column, excluding it from features."""
    dataset = ds.AutoDataset(response_name=None)
    x_train, y_train = dataset.train_data
    # last numeric column is "origin"
    assert dataset.response_name == "origin"
    assert dataset.predictor_names is not None
    assert "origin" not in dataset.predictor_names
    assert len(dataset.predictor_names) == 7


def test_auto_rejects_non_numeric_response() -> None:
    """A non-numeric or unknown response column is rejected."""
    for bad in ("name", "nonsense"):
        with pytest.raises(ValueError):
            ds.AutoDataset(response_name=bad).train_data


def _assert_no_nulls(x: pd.DataFrame, y: pd.Series) -> None:
    """Assert a feature frame and its labels contain no missing values."""
    assert not np.any(x.isna().to_numpy())
    assert not np.any(y.isna().to_numpy())


def test_advertising_has_no_nulls() -> None:
    """Advertising data has no missing values in either split."""
    dataset = ds.AdvertisingDataset()
    _assert_no_nulls(*dataset.train_data)
    _assert_no_nulls(*dataset.test_data)


def test_injection_has_no_nulls() -> None:
    """Injection molding data has no missing values in either split."""
    dataset = ds.InjectionMoldingDataset()
    _assert_no_nulls(*dataset.train_data)
    _assert_no_nulls(*dataset.test_data)


def test_auto_drops_missing_values() -> None:
    """Auto data contains '?' (NaN) cells; rows with missing values are dropped from both splits."""
    dataset = ds.AutoDataset(split=0.8)
    x_train, y_train = dataset.train_data
    x_test, y_test = dataset.test_data
    _assert_no_nulls(x_train, y_train)
    _assert_no_nulls(x_test, y_test)


def test_load_returns_self_and_is_cached() -> None:
    """load() is idempotent and returns the dataset for chaining."""
    dataset = ds.AdvertisingDataset()
    assert dataset.load() is dataset
    # loading twice must not raise and must not re-split data
    first = dataset.train_data
    assert dataset.load() is dataset
    x_train, y_train = dataset.train_data
    assert first[0].equals(x_train)
    assert first[1].equals(y_train)


if __name__ == "__main__":
    pytest.main([__file__])
