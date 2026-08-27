"""Tests for the public analysis API and range utilities."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLSResults

from ml_tools import MAC, MetricSummary, ModelAdequacyChecker, ProblematicSampleMasks
from ml_tools.mac.analyze import DiagnosticPlots
from ml_tools.mac.config import MACConfig
from ml_tools.mac.range import Range


def _make_data(n=100, p=3, seed=0):
    """Generate a linear regression problem with known coefficients."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    beta = np.array([1.0, -2.0, 0.5][:p])
    y = x @ beta + rng.normal(size=n)
    return x, y


def _fit_sklearn(x, y):
    return LinearRegression().fit(x, y)


def test_analyze_sklearn_no_plot() -> None:
    """analyze_sklearn returns metrics, masks and unrendered plotters."""
    x, y = _make_data()
    metric, masks, plots = ModelAdequacyChecker().analyze_sklearn(
        x, y, _fit_sklearn(x, y), plot=False
    )
    assert isinstance(metric, MetricSummary)
    assert isinstance(masks, ProblematicSampleMasks)
    assert isinstance(plots, DiagnosticPlots)
    assert masks.combined.shape == (x.shape[0],)
    assert len(metric.standardized_residuals) == x.shape[0]
    # degrees of freedom = n_features + intercept
    assert metric.dof == x.shape[1] + 1


def test_analyze_mac_alias() -> None:
    """The MAC alias is the same checker class."""
    assert MAC is ModelAdequacyChecker
    model = MAC(config=MACConfig(t_threshold=3.0))
    assert model is not None


def test_masks_flags_track_thresholds() -> None:
    """Raising thresholds relaxes the masks; tightening flags more points."""
    x, y = _make_data()
    model = _fit_sklearn(x, y)
    checker = ModelAdequacyChecker(config=MACConfig(t_threshold=0.5, cook_distance_threshold=0.001))
    _, tight, _ = checker.analyze_sklearn(x, y, model, plot=False)
    _, loose, _ = ModelAdequacyChecker(
        config=MACConfig(t_threshold=1e9, cook_distance_threshold=1e9)
    ).analyze_sklearn(x, y, model, plot=False)
    assert tight.combined.sum() >= loose.combined.sum()
    assert loose.outliers.sum() == 0


def test_plot_all_renders_all_six_figures() -> None:
    """DiagnosticPlots.plot_all creates exactly the six diagnostic figures."""
    x, y = _make_data()
    metric, masks, plots = ModelAdequacyChecker().analyze_sklearn(
        x, y, _fit_sklearn(x, y), plot=False
    )
    open_before = set(plt.get_fignums())
    plots.plot_all(masks)
    created = set(plt.get_fignums()) - open_before
    assert len(created) == 6
    plt.close("all")


def test_plot_all_is_idempotent_and_exposes_figures() -> None:
    """Repeated plot_all calls reuse the cached figures instead of re-rendering."""
    x, y = _make_data()
    metric, masks, plots = ModelAdequacyChecker().analyze_sklearn(
        x, y, _fit_sklearn(x, y), plot=False
    )
    assert plots.figures == {}
    first = plots.plot_all(masks)
    second = plots.plot_all(masks)
    for name in first:
        assert first[name][0] is second[name][0]
    assert set(plots.figures) == set(first)
    assert plots.residuals.figure is first["residuals"][0]
    assert plots.residuals.axes is first["residuals"][1]
    plt.close("all")


def test_plot_true_renders_immediately() -> None:
    """With plot=True the figures are created during analyze."""
    x, y = _make_data()
    open_before = set(plt.get_fignums())
    ModelAdequacyChecker().analyze_sklearn(x, y, _fit_sklearn(x, y), plot=True)
    created = set(plt.get_fignums()) - open_before
    assert len(created) == 6
    plt.close("all")


def test_plot_false_creates_no_figures() -> None:
    """With plot=False no figures are created at all (lazy)."""
    x, y = _make_data()
    open_before = set(plt.get_fignums())
    ModelAdequacyChecker().analyze_sklearn(x, y, _fit_sklearn(x, y), plot=False)
    assert set(plt.get_fignums()) == open_before


@pytest.fixture(name="statsmodels_data")
def _statsmodels_data() -> tuple[np.ndarray, np.ndarray, OLSResults]:
    """Fitted statsmodels OLS model on an intercept-augmented design matrix."""
    x, y = _make_data()
    result = sm.OLS(y, sm.add_constant(x)).fit()
    return sm.add_constant(x), y, result


def test_analyze_statsmodels(statsmodels_data) -> None:
    """analyze_statsmodels matches the sklearn path on the same model."""
    x_const, y, result = statsmodels_data
    metric, masks, plots = ModelAdequacyChecker().analyze_statsmodels(
        x_const, y, result, plot=False
    )
    # the constant was stripped from the features
    assert metric.n_samples == y.shape[0]
    assert metric.dof == x_const.shape[1]
    assert masks.outliers.shape == (y.shape[0],)
    # metrics agree with the sklearn fit of the identical model
    x = x_const[:, 1:]
    metric_sk, _, _ = ModelAdequacyChecker().analyze_sklearn(x, y, _fit_sklearn(x, y), plot=False)
    assert np.isclose(metric.r_squared, metric_sk.r_squared)
    assert np.isclose(metric.rse, metric_sk.rse)
    assert isinstance(plots, DiagnosticPlots)


def test_analyze_statsmodels_no_intercept(statsmodels_data) -> None:
    """analyze_statsmodels works for models fitted without an intercept."""
    _, y, _ = statsmodels_data
    x = np.random.default_rng(0).normal(size=(y.shape[0], 2))
    result = sm.OLS(y, x).fit()
    metric, masks, _ = ModelAdequacyChecker().analyze_statsmodels(x, y, result, plot=False)
    assert metric.dof == 2
    # RSE should match the statsmodels residual standard error
    assert np.isclose(metric.rse, np.sqrt(result.scale))


def test_range_diff_and_quantile() -> None:
    """Range summarises min/max, diff and quantiles correctly."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    r = Range(values)
    assert r.diff() == 3.0
    lo, hi = r.range()
    assert (lo, hi) == (1.0, 4.0)
    assert np.isclose(r.quantile(0.0), 1.0)
    assert np.isclose(r.quantile(1.0), 4.0)
    assert np.isclose(r.quantile(0.5), 2.5)
    # multiple quantiles
    q = r.quantile(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(q, [1.0, 2.5, 4.0])


def test_range_padded_and_bounded() -> None:
    """Padding widens the range and bounds clip it."""
    r = Range(np.array([0.0, 10.0]), padding=0.1)
    lo, hi = r.padded_range()
    assert lo < 0.0
    assert hi > 10.0
    lo_c, hi_c = r.padded_range(bounds=(1.0, 9.0))
    assert (lo_c, hi_c) == (1.0, 9.0)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
