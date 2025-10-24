"""Utilities for value ranges."""

import numpy as np


class Range:
    """Class for summarizing value ranges along a specified axis."""

    def __init__(self, values: np.ndarray, axis=0, padding: float = 0.05) -> None:
        """Initialize.

        Args:
            values: Array of values to summarize.
            axis: Axis along which to compute min and max.
            padding: Fractional padding to apply to the range.
        """
        self._axis = axis
        self._min = values.min(axis=self._axis)
        self._max = values.max(axis=self._axis)
        self._padding = padding

    def diff(self) -> np.ndarray:
        """Get the difference between max and min."""
        return self._max - self._min

    def range(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the (min, max) range."""
        return self._min, self._max

    def padded_range(
        self, padding: float | None = None, bounds: tuple[float, float] | None = None
    ) -> np.ndarray:
        """Get the padded range which is limited by bounds."""
        if padding is None:
            padding = self._padding
        diff = self.diff()
        borders = np.array([self._min - padding * diff, self._max + padding * diff])
        if bounds is not None:
            borders = np.clip(borders, bounds[0], bounds[1])
        return borders

    def quantile(self, q: np.ndarray) -> np.ndarray:
        """Get the quantile value(s) within the range."""
        return self._min + q * self.diff()
