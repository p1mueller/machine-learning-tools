import abc
from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ml_tools.mac.fit import FitSummary


class Metric:
    @property
    def name(self):
        return self.__class__.__name__

    @cached_property
    def value(self):
        return self._compute()

    @abc.abstractmethod
    def _compute(self):
        """Compute the metric value."""
        pass


class Leverage(Metric):
    def __init__(self, data: np.ndarray, add_intercept: bool = True) -> None:
        self.data = data
        self.add_intercept = add_intercept

    def _compute(self) -> float:
        x = self.data
        if self.add_intercept:
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        leverage = (x * np.linalg.pinv(x).T).sum(axis=1)
        return leverage


class RSS(Metric):
    def __init__(self, residuals: np.ndarray) -> None:
        self.residuals = residuals

    def _compute(self) -> float:
        rss = np.sum(self.residuals**2)
        return rss.item()


class RSESquared(Metric):
    def __init__(self, rss: RSS, dof: int) -> None:
        self._metric = rss
        self.dof = dof

    @classmethod
    def from_residuals(cls, residuals: np.ndarray, dof: int) -> "RSESquared":
        rss_metric = RSS(residuals)
        return cls(rss_metric, dof)

    @property
    def residuals(self) -> np.ndarray:
        return self._metric.residuals

    @property
    def rss(self) -> float:
        return self._metric.value

    def _compute(self) -> float:
        n_samples = self.residuals.shape[0]
        rss = np.sum(self.residuals**2)
        rse2 = rss / (n_samples - self.dof)
        return rse2.item()


class RSE(Metric):
    def __init__(self, rse2_metric: RSESquared) -> None:
        self._metric = rse2_metric

    @classmethod
    def from_residuals(cls, residuals: np.ndarray, dof: int) -> "RSE":
        return cls(RSESquared.from_residuals(residuals, dof))

    @property
    def rss(self) -> float:
        return self._metric.rss

    @property
    def dof(self) -> int:
        return self._metric.dof

    @property
    def residuals(self) -> np.ndarray:
        return self._metric.residuals

    @cached_property
    def squared_value(self) -> float:
        return self._metric.value

    def _compute(self) -> float:
        rse2 = self._metric.value
        return np.sqrt(rse2).item()


class StandardizedResiduals(Metric):
    def __init__(self, rse_metric: RSE, leverage: Leverage) -> None:
        self._rse = rse_metric
        self._leverage = leverage

    @classmethod
    def from_residuals(
        cls,
        residuals: np.ndarray,
        dof: int,
        data: np.ndarray,
        add_intercept: bool = True,
    ) -> "StandardizedResiduals":
        rse_metric = RSE.from_residuals(residuals, dof)
        leverage_metric = Leverage(data, add_intercept)
        return cls(rse_metric, leverage_metric)

    @property
    def residuals(self) -> np.ndarray:
        return self._rse.residuals

    @property
    def rss(self) -> float:
        return self._rse.rss

    @property
    def rse(self) -> float:
        return self._rse.value

    @property
    def leverage(self) -> np.ndarray:
        return self._leverage.value

    @property
    def dof(self) -> int:
        return self._rse.dof

    def _compute(self) -> np.ndarray:
        denom = self.rse * np.sqrt(1 - self.leverage)
        denom += 1e-12  # to avoid division by zero
        standardized = self.residuals / denom
        return standardized


class CooksDistance(Metric):
    def __init__(self, standardized_residuals: StandardizedResiduals) -> None:
        self._std_resid = standardized_residuals

    @classmethod
    def from_residuals(
        cls,
        residuals: np.ndarray,
        dof: int,
        data: np.ndarray,
        add_intercept: bool = True,
    ) -> "CooksDistance":
        std_resid_metric = StandardizedResiduals.from_residuals(residuals, dof, data, add_intercept)
        return cls(std_resid_metric)

    @property
    def standardized_residuals(self) -> np.ndarray:
        return self._std_resid.value

    @property
    def leverage(self) -> np.ndarray:
        return self._std_resid.leverage

    @property
    def dof(self) -> int:
        return self._std_resid.dof

    @property
    def rss(self) -> float:
        return self._std_resid.rss

    @property
    def rse(self) -> float:
        return self._std_resid.rse

    def _compute(self) -> np.ndarray:
        return cooks_distance(self._std_resid.value, self._std_resid.leverage, self._std_resid.dof)


class VarianceInflectionFactor(Metric):
    def __init__(self, data: np.ndarray) -> None:
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
    def __init__(self, residuals: np.ndarray, offset: int = 1) -> None:
        self._residuals = residuals
        self._offset = offset

    def _compute(self) -> float:
        return correlation(self._residuals[: -self._offset], self._residuals[self._offset :])


class MetricSummary:
    """Summary of various regression metrics."""

    def __init__(self, summary: FitSummary):
        self._summary = summary
        self.cook_metric = CooksDistance.from_residuals(
            summary.residuals, summary.dof, summary.x, summary.has_intercept
        )
        self.vif_metric = VarianceInflectionFactor(summary.x)
        self._residual_correlation = ResidualCorrelation(summary.residuals)

    @property
    def n_samples(self) -> int:
        return self._summary.n_samples

    @property
    def dof(self) -> int:
        return self._summary.dof

    @property
    def leverage(self) -> np.ndarray:
        return self.cook_metric.leverage

    @property
    def rss(self) -> float:
        return self.cook_metric.rss

    @property
    def rse(self) -> float:
        return self.cook_metric.rse

    @property
    def leverage(self) -> np.ndarray:
        return self.cook_metric.leverage

    @property
    def standardized_residuals(self) -> np.ndarray:
        return self.cook_metric.standardized_residuals

    @property
    def cooks_distance(self) -> np.ndarray:
        return self.cook_metric.value

    @property
    def residual_correlation(self) -> float:
        return self._residual_correlation.value

    @property
    def vif(self) -> np.ndarray:
        return self.vif_metric.value

    def pretty_vif(self, predictor_names: list[str] | None = None) -> str:
        if predictor_names is None:
            predictor_names = self._summary.predictor_names
        df = pd.DataFrame(self.vif, columns=["VIF"], index=predictor_names)
        return df


def cooks_distance(
    normed_residuals: np.ndarray,
    leverage: np.ndarray,
    dof: int,
) -> np.ndarray:
    cook_dist = normed_residuals**2 / dof * leverage / (1 - leverage + 1e-12)
    return cook_dist


def standardized_residuals_from_cook(
    cook_distance: np.ndarray,
    leverage: np.ndarray,
    dof: int,
) -> np.ndarray:
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
