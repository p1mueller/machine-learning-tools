"""Tests for the public analysis API and range utilities."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.linear_model import LinearRegression

from ml_tools import MAC, MetricSummary, ModelAdequacyChecker, ProblematicSampleMasks
from ml_tools.mac.config import MACConfig
from ml_tools.mac.range import Range


def _make_data(n=100, p=3, seed=0):
    """Generate a linear regression problem with known coefficients."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    beta = np.array([1.0, -2.0, 0.5][:p])
    y = x @ beta + rng.normal(size=n)
    return x, y


def test_analyze_sklearn_no_plot() -> None:
    """analyze_sklearn returns a MetricSummary and masks, and does not plot."""
    x, y = _make_data()
    model = LinearRegression().fit(x, y)
    metric, masks = ModelAdequacyChecker().analyze_sklearn(x, y, model, plot=False)
    assert isinstance(metric, MetricSummary)
    assert isinstance(masks, ProblematicSampleMasks)
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
    model = LinearRegression().fit(x, y)
    checker = ModelAdequacyChecker(config=MACConfig(t_threshold=0.5, cook_distance_threshold=0.001))
    _, tight = checker.analyze_sklearn(x, y, model, plot=False)
    loose = ModelAdequacyChecker(
        config=MACConfig(t_threshold=1e9, cook_distance_threshold=1e9)
    ).analyze_sklearn(x, y, model, plot=False)[1]
    assert tight.combined.sum() >= loose.combined.sum()
    assert loose.outliers.sum() == 0


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
