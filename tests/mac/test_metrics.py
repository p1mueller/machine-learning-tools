"""Unit tests for regression diagnostic metrics."""

import numpy as np
import statsmodels.api as sm
from pytest import fixture
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLSResults
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ml_tools.mac.metric import (
    RSE,
    RSS,
    CooksDistance,
    Leverage,
    Metric,
    ResidualCorrelation,
    StandardizedResiduals,
    VarianceInflectionFactor,
)

np.random.seed(42)


def _sample_data(
    n_samples: int = 1000, n_features: int = 3, noise_std: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample data for testing."""
    contributions = np.random.uniform(1.0, 3.0, size=n_features)
    contributions *= np.random.choice([-1, 1], size=n_features)
    noise = np.random.randn(n_samples)
    noise *= noise_std
    x = np.random.randn(n_samples, n_features)
    y = x @ contributions + noise
    return x, y, noise


@fixture(name="basic")
def _basic_data():
    return _sample_data()


def _fit_statsmodel(x: np.ndarray, y: np.ndarray) -> OLSResults:
    """Fit an OLS model and return the fitted model."""
    result = sm.OLS(y, x).fit()
    return result


def _fit_sklearn(x: np.ndarray, y: np.ndarray):
    model = LinearRegression().fit(x, y)
    return model


def _get_residuals(data) -> np.ndarray:
    x, y, _ = data
    result = _fit_statsmodel(x, y)
    residuals = result.resid
    return residuals, result


def test_metric_base():
    """Test that Metric base class caches computed value."""
    computed = 0

    class Variance(Metric):
        def __init__(self, values: np.ndarray):
            self._values = values

        def _compute(self) -> float:
            nonlocal computed
            computed += 1
            return self._values.var()

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    var = Variance(data)
    assert computed == 0
    assert np.isclose(var.value, data.var())
    assert computed == 1
    var.value  # access again
    assert computed == 1  # should not recompute


def test_leverage(basic):
    """Test Leverage metric against statsmodels OLS implementation."""
    x, y, _ = basic
    leverage = Leverage(x, add_intercept=False)
    result = _fit_statsmodel(x, y)
    expected_leverage = result.get_influence().hat_matrix_diag
    assert np.allclose(leverage.value, expected_leverage)

    x_bias = sm.add_constant(x)
    leverage_bias = Leverage(x, add_intercept=True)
    model_bias = _fit_statsmodel(x_bias, y)
    expected_leverage_bias = model_bias.get_influence().hat_matrix_diag
    assert np.allclose(leverage_bias.value, expected_leverage_bias)


def test_rss(basic):
    """Test RSS metric against statsmodels OLS implementation."""
    residuals, result = _get_residuals(basic)
    metric = RSS(residuals)
    expected_rss = result.mse_resid * result.df_resid
    assert np.isclose(metric.value, expected_rss)


def test_rse(basic):
    """Test RSE metric against statsmodels OLS implementation."""
    residuals, result = _get_residuals(basic)
    metric = RSE.from_residuals(residuals, dof=result.df_model)
    expected_rse = np.sqrt(result.scale)
    assert np.isclose(metric.value, expected_rse)


def test_standardized_residuals(basic):
    """Test Standardized Residuals metric against statsmodels OLS implementation."""
    residuals, result = _get_residuals(basic)
    metric = StandardizedResiduals.from_residuals(
        residuals, dof=result.df_model, data=basic[0], add_intercept=False
    )
    expected_value = result.get_influence().resid_studentized_internal
    assert np.allclose(metric.value, expected_value)

    b2 = (sm.add_constant(basic[0]), *basic[1:])
    r2, res2 = _get_residuals(b2)
    dof = res2.nobs - res2.df_resid
    metric = StandardizedResiduals.from_residuals(r2, dof=dof, data=basic[0], add_intercept=True)
    expected_value = res2.get_influence().resid_studentized
    assert np.allclose(metric.value, expected_value)


def test_cooks_distance(basic):
    """Test Standardized Residuals metric against statsmodels OLS implementation."""
    residuals, result = _get_residuals(basic)
    metric = CooksDistance.from_residuals(
        residuals, dof=result.df_model, data=basic[0], add_intercept=False
    )
    expected_value = result.get_influence().cooks_distance[0]
    assert np.allclose(metric.value, expected_value), (metric.value - expected_value)[:10]

    b2 = (sm.add_constant(basic[0]), *basic[1:])
    r2, res2 = _get_residuals(b2)
    dof = res2.nobs - res2.df_resid
    metric = CooksDistance.from_residuals(r2, dof=dof, data=basic[0], add_intercept=True)
    expected_value = res2.get_influence().cooks_distance[0]
    assert np.allclose(metric.value, expected_value)


def test_variance_inflation_factor(basic):
    """Test Variance Inflation Factor metric against statsmodels implementation."""
    x = basic[0]
    metric = VarianceInflectionFactor(x)
    x_bias = sm.add_constant(x)
    expected_value = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        expected_value[i] = variance_inflation_factor(x_bias, i + 1)
    assert np.allclose(metric.value, expected_value)

    metric = VarianceInflectionFactor(np.ones((10, 1)))
    assert np.allclose(metric.value, np.array([1.0]))


def test_residual_correlation(basic):
    """Test Residual Correlation metric against numpy implementation."""
    residuals, _ = _get_residuals(basic)
    metric = ResidualCorrelation(residuals, lag=1)
    expected_value = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    assert np.isclose(metric.value, expected_value)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
