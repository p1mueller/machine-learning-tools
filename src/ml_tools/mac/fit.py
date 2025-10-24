"""Data model for summarizing fitted regression models."""

from functools import cached_property

import numpy as np
import pydantic_numpy.typing as pdn
from pydantic import BaseModel, Field


class FitSummary(BaseModel):
    """Summary of a fitted regression model."""

    x: pdn.Np2DArray = Field(description="Input features")
    y_true: pdn.Np1DArray = Field(description="Target values")
    y_pred: pdn.Np1DArray = Field(description="Predicted target values from the model")
    dof: int = Field(description="Degrees of freedom")
    has_intercept: bool = Field(description="Whether the model includes an intercept term")
    predictor_names: list[str] = Field(description="Names of predictor variables")

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return self.x.shape[0]

    @cached_property
    def residuals(self) -> np.ndarray:
        """Residuals between ground truth and model."""
        return self.y_true - self.y_pred
