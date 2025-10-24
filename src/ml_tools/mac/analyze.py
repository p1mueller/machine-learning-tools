"""MAC main analysis class."""

import numpy as np
from sklearn.linear_model import LinearRegression

import ml_tools.mac.plot as macplot
from ml_tools.mac.config import MACConfig, get_default_config
from ml_tools.mac.detection import ProblematicSampleMasks
from ml_tools.mac.fit import FitSummary
from ml_tools.mac.metric import MetricSummary


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
    ):
        """Analyze a scikit-learn LinearRegression model fit.

        Args:
            x: Feature matrix used for fitting the model.
            y: Ground truth values.
            model: Fitted scikit-learn LinearRegression model.
            y_pred: Predicted target values. If None, will be computed using the model.
            predictor_names: Optional list of predictor names. If None, default names will be used.
            plot: Whether to generate diagnostic plots.
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
            has_intercept=model.fit_intercept,
            predictor_names=predictor_names,
        )
        return self.analyze(summary, plot=plot)

    # TODO:
    # def analyze_statsmodels(self, data: pd.DataFrame, model):
    #     pass

    def analyze(
        self, summary: FitSummary, plot: bool = True
    ) -> tuple[MetricSummary, ProblematicSampleMasks]:
        """Analyze the model fit summary and optionally plot diagnostic plots."""
        metric = MetricSummary(summary)
        masks = ProblematicSampleMasks.from_metric_summary(metric, self._config)

        if plot:
            vif_df = metric.pretty_vif(summary.predictor_names)
            plotters = [
                macplot.TukeyAnscombePlotter(summary.y_pred, summary.residuals, self._config),
                macplot.ScaleLocationPlotter(
                    summary.y_pred, metric.standardized_residuals, self._config
                ),
                macplot.QQPlotter(metric.standardized_residuals, config=self._config),
                macplot.SensitivityPlotter(
                    metric.leverage, metric.cook_metric, config=self._config
                ),
                macplot.ResidualCorrelationPlotter(
                    summary.residuals, metric.residual_correlation, config=self._config
                ),
                macplot.VIFPlotter(vif_df, config=self._config),
            ]
            for plotter in plotters:
                plotter.plot(masks=masks)

        return metric, masks


MAC = ModelAdequacyChecker
