"""Metrics for regression analysis."""

import abc
from functools import cached_property
from typing import Any, override

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ml_tools.mac.fit import FitSummary


class Metric(abc.ABC):
    """Base class for regression metrics."""

    @property
    def name(self):
        """Name of the metric."""
        return self.__class__.__name__

    @cached_property
    def value(self):
        """Compute and cache the metric value."""
        return self._compute()

    @abc.abstractmethod
    def _compute(self) -> Any:
        """Compute the metric value."""


class Leverage(Metric):
    """Compute leverage values for each sample."""

    def __init__(self, data: np.ndarray, add_intercept: bool = True) -> None:
        """Initialize.

        Args:
            data: Input feature matrix.
            add_intercept: Whether to add an intercept term to the data.
        """
        self.data = data
        self.add_intercept = add_intercept

    @override
    def _compute(self) -> np.ndarray:
        x = self.data
        if self.add_intercept:
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        leverage = (x * np.linalg.pinv(x).T).sum(axis=1)
        return leverage


class RSS(Metric):
    """Compute Residual Sum of Squares (RSS) metric."""

    def __init__(self, residuals: np.ndarray) -> None:  # noqa: D107
        self.residuals = residuals

    @property
    def n_samples(self) -> np.ndarray:  # noqa: D102
        return self.residuals.shape[0]

    def _compute(self) -> float:
        rss = np.sum(self.residuals**2)
        return rss.item()


class RSE(Metric):
    """Compute Residual Standard Error (RSE) metric."""

    def __init__(self, rss_metric: RSS, dof: int) -> None:  # noqa: D107
        self._metric = rss_metric
        self.dof = dof

    @classmethod
    def from_residuals(cls, residuals: np.ndarray, dof: int) -> "RSE":
        """Create RSE metric from residuals.

        Args:
            residuals: Residuals from the regression model.
            dof: Degrees of freedom (including bias)
        """
        return cls(RSS(residuals), dof)

    @property
    def rss(self) -> float:  # noqa: D102
        return self._metric.value

    @property
    def residuals(self) -> np.ndarray:  # noqa: D102
        return self._metric.residuals

    def _compute(self) -> float:
        rss = self._metric.value
        rse = np.sqrt(rss / (self._metric.n_samples - self.dof))
        return rse.item()


class StandardizedResiduals(Metric):
    r"""Compute t standardized residuals for each sample.

    Standardized residuals are calculated as:
    $$
    r_i = \frac{e_i}{\hat{\varsigma} \sqrt{1 - h_{ii}}}
    $$
    """

    def __init__(self, rse: RSE, leverage: Leverage) -> None:  # noqa: D107
        self._rse = rse
        self._leverage = leverage

    @classmethod
    def from_residuals(
        cls,
        residuals: np.ndarray,
        dof: int,
        data: np.ndarray,
        add_intercept: bool = True,
    ) -> "StandardizedResiduals":
        """Create Standardized Residuals metric from residuals.

        Args:
            residuals: Residuals from the regression model.
            dof: Degrees of freedom (including bias)
            data: Input feature matrix used in the regression.
            add_intercept: Whether to add an intercept term to the data.
        """
        rse_metric = RSE.from_residuals(residuals, dof)
        leverage_metric = Leverage(data, add_intercept)
        return cls(rse_metric, leverage_metric)

    @property
    def residuals(self) -> np.ndarray:  # noqa: D102
        return self._rse.residuals

    @property
    def rss(self) -> float:  # noqa: D102
        return self._rse.rss

    @property
    def rse(self) -> float:  # noqa: D102
        return self._rse.value

    @property
    def leverage(self) -> np.ndarray:  # noqa: D102
        return self._leverage.value

    @property
    def dof(self) -> int:  # noqa: D102
        return self._rse.dof

    def _compute(self) -> np.ndarray:
        denom = self.rse * np.sqrt(1 - self.leverage)
        denom += 1e-12  # to avoid division by zero
        standardized = self.residuals / denom
        return standardized


class CooksDistance(Metric):
    r"""Compute Cook's Distance for each sample.

    Cook's Distance measures the influence of each data point on the fitted regression model.
    $$
    D_i &= \frac{r_i^2}{p} \cdot \frac{h_{ii}}{1 - h_{ii}} \\
    &= \frac{e_i^2}{p \hat{\varsigma}^2} \cdot \frac{h_{ii}}{(1 - h_{ii})^2}
    $$
    """

    def __init__(self, standardized_residuals: StandardizedResiduals) -> None:  # noqa: D107
        self._std_resid = standardized_residuals

    @classmethod
    def from_residuals(
        cls,
        residuals: np.ndarray,
        dof: int,
        data: np.ndarray,
        add_intercept: bool = True,
    ) -> "CooksDistance":
        """Create Cook's Distance metric from residuals.

        Args:
            residuals: Residuals from the regression model.
            dof: Degrees of freedom (including bias)
            data: Input feature matrix used in the regression.
            add_intercept: Whether to add an intercept term to the data.
        """
        std_resid_metric = StandardizedResiduals.from_residuals(residuals, dof, data, add_intercept)
        return cls(std_resid_metric)

    @property
    def standardized_residuals(self) -> np.ndarray:  # noqa: D102
        return self._std_resid.value

    @property
    def leverage(self) -> np.ndarray:  # noqa: D102
        return self._std_resid.leverage

    @property
    def dof(self) -> int:  # noqa: D102
        return self._std_resid.dof

    @property
    def rss(self) -> float:  # noqa: D102
        return self._std_resid.rss

    @property
    def rse(self) -> float:  # noqa: D102
        return self._std_resid.rse

    def _compute(self) -> np.ndarray:
        return cooks_distance(self._std_resid.value, self._std_resid.leverage, self._std_resid.dof)


class VarianceInflectionFactor(Metric):
    """Compute Variance Inflation Factor (VIF) for each feature.

    An intercept is included implicitly; it does not affect the computed values.
    """

    def __init__(self, data: np.ndarray) -> None:
        """Initialize."""
        self.data = data

    def _compute(self) -> np.ndarray:
        n_features = self.data.shape[1]
        if n_features < 2:
            return np.array([1.0])
        vif = np.zeros(n_features)
        indices = np.arange(n_features)
        for feature in range(n_features):
            mask = indices != feature
            x = self.data[:, mask]
            y = self.data[:, feature]
            model = LinearRegression().fit(x, y)
            r2 = model.score(x, y)
            vif[feature] = 1 / (1 - r2 + 1e-12)
        return vif


class ResidualCorrelation(Metric):
    """Compute the correlation between residuals and their lagged values."""

    def __init__(self, residuals: np.ndarray, lag: int = 1) -> None:
        """Initialize.

        Args:
            residuals: Residuals from the regression model.
            lag: Lag value for computing correlation.
        """
        self._residuals = residuals
        self._lag = lag

    def _compute(self) -> float:
        return correlation(self._residuals[: -self._lag], self._residuals[self._lag :])


class MetricSummary:
    """Summary of various regression metrics."""

    def __init__(self, summary: FitSummary):  # noqa: D107
        self._summary = summary
        self.cook_metric = CooksDistance.from_residuals(
            summary.residuals, summary.dof, summary.x, summary.has_intercept
        )
        self.vif_metric = VarianceInflectionFactor(summary.x)
        self._residual_correlation = ResidualCorrelation(summary.residuals)

    @property
    def n_samples(self) -> int:  # noqa: D102
        return self._summary.n_samples

    @property
    def dof(self) -> int:  # noqa: D102
        return self._summary.dof

    @property
    def rss(self) -> float:  # noqa: D102
        return self.cook_metric.rss

    @property
    def rse(self) -> float:  # noqa: D102
        return self.cook_metric.rse

    @property
    def leverage(self) -> np.ndarray:  # noqa: D102
        return self.cook_metric.leverage

    @property
    def standardized_residuals(self) -> np.ndarray:  # noqa: D102
        return self.cook_metric.standardized_residuals

    @property
    def cooks_distance(self) -> np.ndarray:  # noqa: D102
        return self.cook_metric.value

    @property
    def residual_correlation(self) -> float:  # noqa: D102
        return self._residual_correlation.value

    @property
    def tss(self) -> float:
        """Total Sum of Squares: sum of squared deviations of the response from its mean."""
        y = self._summary.y_true
        y_mean = y.mean()
        return float(np.sum((y - y_mean) ** 2))

    @property
    def r_squared(self) -> float:
        r"""Coefficient of determination $R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}$.

        Uses the centered decomposition (unbiased estimator of the mean
        response). For models fitted without an intercept, the uncentered
        definition $R^2 = 1 - \frac{\text{RSS}}{\sum y^2}$ is used, matching
        statsmodels.
        Returns NaN for a constant response, where R² is undefined.
        """
        if self._summary.has_intercept:
            tss = self.tss
        else:
            tss = float(np.sum(self._summary.y_true**2))
        if tss <= 0:
            return float("nan")
        return 1.0 - self.rss / tss

    @property
    def adj_r_squared(self) -> float:
        r"""Adjusted R²: $1 - \left(1 - R^2\right)\frac{n - 1}{n - p}$.

        `n` is the number of samples and `p` the number of fitted parameters
        (including the intercept).
        """
        n = self.n_samples
        p = self.dof
        if n <= p:
            return float("nan")
        return 1.0 - (1.0 - self.r_squared) * (n - 1) / (n - p)

    @property
    def f_statistic(self) -> float:
        r"""F-statistic for the model: $\frac{\text{SSR}/k}{\text{RSS}/(n - p)}$.

        `k` is the number of slopes (fitted parameters excluding the
        intercept), so that the value matches statsmodels' `fvalue`.
        The explained sum of squares uses the centered decomposition
        (tss - rss) with an intercept and the uncentered one (sum of squared
        fitted values) without.
        Returns NaN when no slopes are fitted, when the residual degrees of
        freedom do not exceed the number of parameters, or when the residual
        sum of squares is zero.
        """
        n = self.n_samples
        p = self.dof
        k = p - 1 if self._summary.has_intercept else p
        if k <= 0 or n <= p or self.rss <= 0:
            return float("nan")
        if self._summary.has_intercept:
            ssr = self.tss - self.rss
        else:
            ssr = float(np.sum(self._summary.y_pred**2))
        return (ssr / k) / (self.rss / (n - p))

    @property
    def vif(self) -> np.ndarray:  # noqa: D102
        return self.vif_metric.value

    def pretty_vif(self, predictor_names: list[str] | None = None) -> pd.DataFrame:
        """Create a pretty DataFrame of VIF values."""
        if predictor_names is None:
            predictor_names = self._summary.predictor_names
        series = pd.Series(self.vif, index=predictor_names, name="VIF")
        return series.to_frame()


def cooks_distance(
    normed_residuals: np.ndarray,
    leverage: np.ndarray,
    dof: int,
) -> np.ndarray:
    """Compute Cook's distance from standardized residuals.

    Args:
        normed_residuals: Standardized residuals.
        leverage: Leverage values.
        dof: Degrees of freedom (including bias)
    """
    cook_dist = normed_residuals**2 / dof * leverage / (1 - leverage + 1e-12)
    return cook_dist


def standardized_residuals_from_cook(
    cook_distance: np.ndarray,
    leverage: np.ndarray,
    dof: int,
) -> np.ndarray:
    """Compute standardized residuals from Cook's distance.

    Args:
        cook_distance: Cook's distance values.
        leverage: Leverage values.
        dof: Degrees of freedom (including bias)
    """
    normed_residuals_squared = cook_distance * dof * (1 - leverage) / (leverage + 1e-12)
    normed_residuals = np.sqrt(normed_residuals_squared)
    return normed_residuals


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient between two 1D arrays."""
    unbiased_x = x - x.mean()
    unbiased_y = y - y.mean()
    cov = np.sum(unbiased_x * unbiased_y)
    var_x = np.sum(unbiased_x**2)
    var_y = np.sum(unbiased_y**2)
    corr = cov / np.sqrt(var_x * var_y + 1e-12)
    return corr.item()
