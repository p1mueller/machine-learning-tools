import numpy as np


class Range:
    def __init__(self, values: np.ndarray, axis=0, padding: float = 0.05) -> None:
        self._axis = axis
        self._min = values.min(axis=self._axis)
        self._max = values.max(axis=self._axis)
        self._padding = padding

    def diff(self) -> np.ndarray:
        return self._max - self._min

    def range(self) -> tuple[np.ndarray, np.ndarray]:
        return self._min, self._max

    def padded_range(
        self, padding: float | None = None, bounds: tuple[float, float] | None = None
    ) -> np.ndarray:
        if padding is None:
            padding = self._padding
        diff = self.diff()
        borders = np.array([self._min - padding * diff, self._max + padding * diff])
        if bounds is not None:
            borders = np.clip(borders, bounds[0], bounds[1])
        return borders

    def quantile(self, q: np.ndarray) -> np.ndarray:
        return self._min + q * self.diff()
