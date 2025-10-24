"""Module for identifying problematic samples in regression analysis."""

from functools import cached_property

import numpy as np
from pydantic import BaseModel, Field
from pydantic_numpy.typing import Np1DArray

from ml_tools.mac.config import MACConfig
from ml_tools.mac.metric import MetricSummary


class ProblematicSampleMasks(BaseModel):
    """Masks identifying problematic samples in regression analysis."""

    outliers: Np1DArray = Field(description="Boolean mask indicating outlier samples")
    high_leverage: Np1DArray = Field(description="Boolean mask indicating high leverage samples")
    influential: Np1DArray = Field(
        description="Boolean mask indicating samples with critical Cook's distance"
    )

    @classmethod
    def from_metric_summary(  # noqa: D102
        cls, metric: MetricSummary, config: MACConfig
    ) -> "ProblematicSampleMasks":
        n_samples = metric.n_samples
        leverage_threshold = config.leverage_threshold_factor * metric.dof / n_samples
        outliers = np.abs(metric.standardized_residuals) > config.t_threshold
        high_leverage = metric.leverage > leverage_threshold
        influential = metric.cooks_distance > config.cook_distance_threshold
        return cls(outliers=outliers, high_leverage=high_leverage, influential=influential)

    @cached_property
    def combined(self) -> Np1DArray:
        """Combined mask of all identified samples."""
        return self.outliers | self.high_leverage | self.influential
