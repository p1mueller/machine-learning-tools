"""Smoke tests that each diagnostic plot renders without error."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import ml_tools.mac.plot as macplot
from ml_tools.mac.config import MACConfig
from ml_tools.mac.detection import ProblematicSampleMasks
from ml_tools.mac.fit import FitSummary
from ml_tools.mac.metric import MetricSummary


def _build(
    n: int = 80, p: int = 3, seed: int = 1
) -> tuple[FitSummary, MetricSummary, ProblematicSampleMasks]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    y = x @ np.array([1.0, -2.0, 0.5][:p]) + rng.normal(size=n)
    model = LinearRegression().fit(x, y)
    summary = FitSummary(
        x=x,
        y_true=y,
        y_pred=model.predict(x),
        dof=p + 1,
        has_intercept=True,
        predictor_names=[f"x{i}" for i in range(p)],
    )
    metric = MetricSummary(summary)
    masks = ProblematicSampleMasks.from_metric_summary(metric, MACConfig())
    return summary, metric, masks


@pytest.fixture(name="built")
def _built() -> tuple[FitSummary, MetricSummary, ProblematicSampleMasks]:
    return _build()


def _render(plotter: macplot.Plotter, masks: ProblematicSampleMasks) -> None:
    """Plot, force rendering, and assert the axes have artists."""
    fig, ax = plotter.plot(masks=masks)
    fig.canvas.draw()  # force rendering so lazy errors surface
    assert ax.get_children()


def test_tukey_anscombe_plot(
    built: tuple[FitSummary, MetricSummary, ProblematicSampleMasks],
) -> None:
    """Render the Tukey-Anscombe (residuals vs fitted) plot."""
    summary, _, masks = built
    _render(macplot.TukeyAnscombePlotter(summary.y_pred, summary.residuals), masks)


def test_scale_location_plot(
    built: tuple[FitSummary, MetricSummary, ProblematicSampleMasks],
) -> None:
    """Render the scale-location plot."""
    summary, metric, masks = built
    _render(macplot.ScaleLocationPlotter(summary.y_pred, metric.standardized_residuals), masks)


def test_qq_plot(built: tuple[FitSummary, MetricSummary, ProblematicSampleMasks]) -> None:
    """Render the normal Q-Q plot."""
    _, metric, masks = built
    _render(macplot.QQPlotter(metric.standardized_residuals), masks)


def test_sensitivity_plot(built: tuple[FitSummary, MetricSummary, ProblematicSampleMasks]) -> None:
    """Render the sensitivity (leverage vs standardized residual) plot."""
    _, metric, masks = built
    _render(macplot.SensitivityPlotter(metric.leverage, metric.cook_metric), masks)


def test_residual_correlation_plot(
    built: tuple[FitSummary, MetricSummary, ProblematicSampleMasks],
) -> None:
    """Render the residuals-vs-index (autocorrelation) plot."""
    summary, metric, masks = built
    _render(
        macplot.ResidualCorrelationPlotter(summary.residuals, metric.residual_correlation), masks
    )


def test_vif_plot(built: tuple[FitSummary, MetricSummary, ProblematicSampleMasks]) -> None:
    """Render the variance inflation factor bar plot."""
    _, metric, masks = built
    _render(macplot.VIFPlotter(metric.pretty_vif()), masks)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
