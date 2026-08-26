"""MAC main analysis class."""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LinearRegression

import ml_tools.mac.plot as macplot
from ml_tools.mac.config import MACConfig, get_default_config
from ml_tools.mac.detection import ProblematicSampleMasks
from ml_tools.mac.fit import FitSummary
from ml_tools.mac.metric import MetricSummary


class DiagnosticPlots(BaseModel):
    """Diagnostic plotters created by :meth:`ModelAdequacyChecker.analyze`.

    Plotters are constructed but not rendered, so figures can be created
    later, where and when needed. Call :meth:`plot_all` to render them all.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    residuals: macplot.Plotter
    scale_location: macplot.Plotter
    qq: macplot.Plotter
    sensitivity: macplot.Plotter
    residual_correlation: macplot.Plotter
    vif: macplot.Plotter

    def plot_all(self, masks: ProblematicSampleMasks | None = None) -> None:
        """Render every diagnostic plot.

        Args:
            masks: Optional problem point masks to highlight in the plots.
        """
        for plotter in self.model_dump().values():
            plotter.plot(masks=masks)


class ModelAdequacyChecker:
    """Performs model adequacy checking for linear regression models."""

    def __init__(self, config: MACConfig | None = None):  # noqa: D107
        if config is None:
            config = get_default_config()
        self._config = config

    def analyze_sklearn(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model: LinearRegression,
        y_pred: np.ndarray | None = None,
        predictor_names: list[str] | None = None,
        plot: bool = True,
    ) -> tuple[MetricSummary, ProblematicSampleMasks, DiagnosticPlots]:
        """Analyze a scikit-learn LinearRegression model fit.

        Args:
            x: Feature matrix used for fitting the model.
            y: Ground truth values.
            model: Fitted scikit-learn LinearRegression model.
            y_pred: Predicted target values. If None, will be computed using the model.
            predictor_names: Optional list of predictor names. If None, default names will be used.
            plot: Whether to render the diagnostic plots immediately. The
                plotters are always returned unrendered in the
                :class:`DiagnosticPlots` result.

        """
        if predictor_names is None:
            predictor_names = [f"x{i}" for i in range(x.shape[1])]
        if y_pred is None:
            y_pred = model.predict(x)
        dof = len(model.coef_) + int(model.fit_intercept)
        summary = FitSummary(
            x=x,
            y_true=y,
            y_pred=y_pred,
            dof=dof,
            has_intercept=bool(model.fit_intercept),
            predictor_names=predictor_names,
        )
        return self.analyze(summary, plot=plot)

    def analyze_statsmodels(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        model: Any,
        y_pred: np.ndarray | None = None,
        predictor_names: list[str] | None = None,
        plot: bool = True,
    ) -> tuple[MetricSummary, ProblematicSampleMasks, DiagnosticPlots]:
        """Analyze a statsmodels OLS model fit.

        ``x`` may include an explicit constant (intercept) column, which is
        detected automatically.

        Args:
            x: Feature matrix used for fitting the model (columns in fit order).
            y: Ground truth values.
            model: Fitted statsmodels OLS model or ``OLSResults``.
            y_pred: Predicted target values. If None, ``model.fittedvalues`` is used.
            predictor_names: Optional list of predictor names. If None, the
                columns of ``x`` (excluding the constant, if present) are used.
            plot: Same as in :meth:`analyze_sklearn`.

        """
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        is_const = np.all(x_arr == 1.0, axis=0)
        has_intercept = bool(is_const.any())
        n_columns = x_arr.shape[1]
        fallback_names = [f"x{i}" for i in range(n_columns)]
        columns = getattr(x, "columns", None)
        column_names = [str(c) for c in (columns if columns is not None else fallback_names)]
        kept = [i for i in range(n_columns) if not is_const[i]]
        if predictor_names is None:
            names = column_names if len(kept) == n_columns else [column_names[i] for i in kept]
        else:
            names = list(predictor_names)
        if y_pred is None:
            y_pred = model.fittedvalues
        summary = FitSummary(
            x=x_arr[:, kept] if has_intercept else x_arr,
            y_true=np.asarray(y, dtype=float).ravel(),
            y_pred=np.asarray(y_pred, dtype=float).ravel(),
            dof=len(kept) + int(has_intercept),
            has_intercept=has_intercept,
            predictor_names=names,
        )
        return self.analyze(summary, plot=plot)

    def analyze(
        self, summary: FitSummary, plot: bool = True
    ) -> tuple[MetricSummary, ProblematicSampleMasks, DiagnosticPlots]:
        """Analyze the model fit summary and create diagnostic plots.

        Args:
            summary: A summary of the fitted model.
            plot: Whether to render the diagnostic plots immediately. If False,
                use the returned :class:`DiagnosticPlots` to render them later.

        Example:
            >>> metric, masks, plots = checker.analyze(summary, plot=False)  # doctest: +SKIP
            >>> plots.plot_all(masks)  # render when and where needed  # doctest: +SKIP
            >>> plt.show()  # doctest: +SKIP
        """
        metric = MetricSummary(summary)
        masks = ProblematicSampleMasks.from_metric_summary(metric, self._config)
        plots = DiagnosticPlots(**self._make_plotters(summary, metric))

        if plot:
            plots.plot_all(masks)

        return metric, masks, plots

    def _make_plotters(
        self, summary: FitSummary, metric: MetricSummary
    ) -> dict[str, macplot.Plotter]:
        vif_df = metric.pretty_vif(summary.predictor_names)
        return {
            "residuals": macplot.TukeyAnscombePlotter(
                summary.y_pred, summary.residuals, self._config
            ),
            "scale_location": macplot.ScaleLocationPlotter(
                summary.y_pred, metric.standardized_residuals, self._config
            ),
            "qq": macplot.QQPlotter(metric.standardized_residuals, config=self._config),
            "sensitivity": macplot.SensitivityPlotter(
                metric.leverage, metric.cook_metric, config=self._config
            ),
            "residual_correlation": macplot.ResidualCorrelationPlotter(
                summary.residuals, metric.residual_correlation, config=self._config
            ),
            "vif": macplot.VIFPlotter(vif_df, config=self._config),
        }


MAC = ModelAdequacyChecker
