"""Plotting utilities for model adequacy."""

import abc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
from labellines import labelLines
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.smoothers_lowess import lowess

from ml_tools.mac.config import Color, MACConfig, get_default_config
from ml_tools.mac.detection import ProblematicSampleMasks
from ml_tools.mac.metric import CooksDistance, standardized_residuals_from_cook
from ml_tools.mac.range import Range


def _compute_color_values(
    values: np.ndarray,
    cmap: plt.Colormap,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Compute color values for the given data values."""
    normalizer = Normalize(vmin, vmax)
    normalized_values = normalizer(values)
    colors = cmap(normalized_values)
    return colors


class Plotter(abc.ABC):
    """Abstract base class for plotters."""

    def __init__(self, config: MACConfig | None) -> None:
        """Initialize."""
        if config is None:
            config = get_default_config()
        self._config = config

    @abc.abstractmethod
    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the data on the given Axes object."""
        pass

    def _determine_figure_size(self, config: MACConfig) -> tuple[float, float]:
        """Determine figure size based on config."""
        return (config.base_width, config.base_height)

    def plot(
        self, ax: plt.Axes | None = None, masks: ProblematicSampleMasks | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the data using Matplotlib.

        Args:
            ax: Optional Matplotlib Axes object to plot on.
                If None, a new figure and axes will be created.
            masks: Masks to highlight problematic samples in plot.
        """
        if ax is None:
            figsize = self._determine_figure_size(self._config)
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        self._plot(ax, masks=masks)
        if self._config.grid_show:
            ax.grid(True, alpha=self._config.grid_alpha)
            ax.set_axisbelow(self._config.grid_below)
        assert isinstance(fig, plt.Figure)
        return fig, ax


class ScatterPlotter(Plotter):
    """Plotter for scatter plots."""

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: MACConfig | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(config)
        self._x = x
        self._y = y

    def _masked_scatter(
        self,
        mask: np.ndarray,
        ax: plt.Axes,
        color: Color,
        edgecolor: Color,
        label: str | None = None,
        annotate: bool = False,
    ) -> None:
        """Internal method to plot the scatter plot on the given Axes object."""
        x = self._x[mask]
        y = self._y[mask]
        ax.scatter(x, y, marker=self._config.marker, color=color, edgecolor=edgecolor, label=label)
        if annotate:
            indices = np.where(mask)[0]
            for i, x_val, y_val in zip(indices, x, y):
                ax.annotate(
                    f"{i}",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(-10, 0),
                    ha="center",
                    fontsize=8,
                    color=edgecolor,
                )

    def _create_masked_scatter_plots(self, ax: plt.Axes, masks: ProblematicSampleMasks) -> None:
        self._masked_scatter(
            ~masks.combined, ax, self._config.colors.marker, self._config.colors.marker_edge
        )
        for label, mask, color, edgecolor in [
            (
                "High Leverage",
                masks.high_leverage,
                self._config.colors.high_leverage,
                self._config.colors.high_leverage_edge,
            ),
            (
                "Influential",
                masks.influential,
                self._config.colors.influential,
                self._config.colors.influential_edge,
            ),
            (
                "Outlier",
                masks.outliers,
                self._config.colors.outlier,
                self._config.colors.outlier_edge,
            ),
        ]:
            self._masked_scatter(mask, ax, color, edgecolor, label=label, annotate=True)
        ax.legend()

    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the scatter plot on the given Axes object."""
        if masks is None:
            ax.scatter(
                self._x,
                self._y,
                marker=self._config.marker,
                color=self._config.colors.marker,
                edgecolor=self._config.colors.marker_edge,
            )
        else:
            self._create_masked_scatter_plots(ax, masks)

        if self.title is not None:
            ax.set_title(self.title)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)


class TukeyAnscombePlotter(ScatterPlotter):
    """Plotter for residuals."""

    title: str = "Tukey-Anscombe Plot"
    xlabel: str = r"Fitted Values $\hat{y}_i$"
    ylabel: str = r"Residuals $e_i$"
    target_line: float | None = 0.0

    def __init__(
        self, fitted_values: np.ndarray, residuals: np.ndarray, config: MACConfig | None = None
    ) -> None:
        """Initialize."""
        super().__init__(fitted_values, residuals, config)

    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the residuals on the given Axes object."""
        super()._plot(ax, masks)
        if self.target_line is not None:
            ax.axhline(self.target_line, color=self._config.colors.target, linestyle="--")
        lowess = _apply_lowess(self._x, self._y, self._config)
        ax.plot(lowess[:, 0], lowess[:, 1], color=self._config.colors.highlight_edge)


class StandardizedResidualPlotter(TukeyAnscombePlotter):
    """Plotter for standardized residuals."""

    title: str = "Standardized Residuals Plot"
    ylabel: str = r"Standardized Residuals $r_i$"

    def __init__(
        self,
        fitted_values: np.ndarray,
        standardized_residuals: np.ndarray,
        config: MACConfig | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(fitted_values, standardized_residuals, config)

    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the standardized residuals on the given Axes object."""
        super()._plot(ax, masks)
        t = self._config.t_threshold
        ax.axhline(-t, color=self._config.colors.outlier_edge, linestyle="--")
        ax.axhline(t, color=self._config.colors.outlier_edge, linestyle="--")


class ScaleLocationPlotter(TukeyAnscombePlotter):
    """Plotter for Scale-Location plot."""

    title: str = "Scale-Location Plot"
    ylabel: str = r"$\sqrt{|r_i|}$"
    target_line: float | None = None

    def __init__(
        self,
        fitted_values: np.ndarray,
        standardized_residuals: np.ndarray,
        config: MACConfig | None = None,
    ) -> None:
        """Initialize."""
        scale_loc = np.sqrt(np.abs(standardized_residuals))
        super().__init__(fitted_values, scale_loc, config)

    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the Scale-Location on the given Axes object."""
        super()._plot(ax, masks)
        t = np.sqrt(self._config.t_threshold)
        ax.axhline(t, color=self._config.colors.outlier_edge, linestyle="--")
        ax.set_ylim(bottom=0)


class QQPlotter(ScatterPlotter):
    """Plotter for QQ plot."""

    title: str | None = "Quantile-Quantile Plot"
    xlabel: str = "Theoretical Quantiles"
    ylabel: str = "Sample Quantiles"

    def __init__(
        self,
        standardized_residuals: np.ndarray,
        base_distribution: sps.rv_continuous = sps.norm,
        config: MACConfig | None = None,
    ) -> None:
        """Initialize."""
        self._base_distribution = base_distribution

        n_samples = standardized_residuals.shape[0]
        quantiles = np.arange(1, n_samples + 1) / (n_samples + 1)

        sorted_indices = np.argsort(standardized_residuals)
        inverse_sorted_indices = np.argsort(sorted_indices)
        sorted_theoretical_quants = self._base_distribution.ppf(quantiles)
        theoretical_quants = sorted_theoretical_quants[inverse_sorted_indices]
        super().__init__(theoretical_quants, standardized_residuals, config)

    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the QQ plot on the given Axes object."""
        super()._plot(ax, masks)
        ax.axline((0, 0), slope=1, color=self._config.colors.target, linestyle="--")


class SensitivityPlotter(ScatterPlotter):
    """Plotter for visualizing sample sensitivity of the model."""

    title: str = "Sensitivity Plot"
    xlabel: str = r"Leverage $h_{ii}$"
    ylabel: str = r"Standardized Residuals $r_i$"

    def __init__(
        self,
        leverage: np.ndarray,
        cook: CooksDistance,
        n_points: int = 100,
        n_levels: int = 5,
        config: MACConfig | None = None,
    ) -> None:
        """Initialize."""
        standardized_residuals = cook.standardized_residuals
        super().__init__(leverage, standardized_residuals, config)
        self._metric = cook
        self._n_points = n_points
        self._n_levels = n_levels

    def _determine_contour_levels(self) -> np.ndarray:
        cook_range = Range(self._metric.value)
        cook_lims = np.clip(cook_range.range(), 0, self._config.cook_distance_threshold)
        ticker = MaxNLocator(nbins=self._n_levels, min_n_ticks=3)
        levels = ticker.tick_values(cook_lims[0], cook_lims[1])
        if levels[0] < 1e-12:
            levels = levels[1:]
        return np.asarray(levels)

    def _plot_cook_contours(self, ax: plt.Axes) -> np.ndarray:
        leverage_range = Range(self._x)
        leverage_border = leverage_range.padded_range(bounds=(0, 1))

        # Determine value ranges
        levels = self._determine_contour_levels()

        # Generate contour lines
        lower_bound = np.log10(self._metric.dof / self._x.shape[0]) - 1
        x = np.logspace(lower_bound, 0, self._n_points)[:, None]
        y_pos = standardized_residuals_from_cook(levels[None], x, self._metric.dof)
        y = np.concatenate([y_pos, -y_pos], axis=-1)

        # Determine colors for contour lines
        cmap = LinearSegmentedColormap.from_list(
            "cook_contour_cmap",
            [self._config.colors.marker_edge, self._config.colors.influential_edge],
        )
        all_levels = np.tile(levels, 2)
        colors = _compute_color_values(all_levels, cmap, 0, self._config.cook_distance_threshold)

        # Plot contour lines
        for yi, color, level in zip(y.T, colors, all_levels):
            ax.plot(x, yi, color=color, zorder=0, label=f"{level:.3g}")

        # Label contour lines
        lines = ax.get_lines()
        label_lines_x = [float(v) for v in np.atleast_1d(leverage_range.quantile(0.9))]
        labelLines(lines, zorder=1, xvals=label_lines_x, fontsize=8)
        for line in lines:
            line.set_label("_nolegend_")
        return leverage_border

    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the sensitivity plot on the given Axes object."""
        residuals_range = Range(self._y)
        residuals_border = residuals_range.padded_range()
        leverage_border = self._plot_cook_contours(ax)
        ax.axhline(
            -self._config.t_threshold, color=self._config.colors.outlier_edge, linestyle="--"
        )
        ax.axhline(self._config.t_threshold, color=self._config.colors.outlier_edge, linestyle="--")
        leverage_threshold = self._metric.dof / self._x.shape[0]
        leverage_threshold *= self._config.leverage_threshold_factor
        ax.axvline(leverage_threshold, color=self._config.colors.high_leverage_edge, linestyle="--")
        super()._plot(ax, masks)

        ax.set_xlim(tuple(leverage_border.tolist()))
        ax.set_ylim(tuple(residuals_border.tolist()))


class ResidualCorrelationPlotter(TukeyAnscombePlotter):
    """Plotter for Residuals vs Index plot."""

    title: str = "Correlation of Residuals"
    xlabel: str = "Index $i$"
    ylabel: str = r"Residuals $e_i$"

    def __init__(
        self,
        residuals: np.ndarray,
        correlation: float,
        time: np.ndarray | None = None,
        config: MACConfig | None = None,
    ) -> None:
        """Initialize."""
        if time is None:
            time = np.arange(len(residuals))
        super().__init__(time, residuals, config)
        self._correlation = correlation

    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the residuals vs index on the given Axes object."""
        super()._plot(ax, masks)
        ax.set_title(self.title + r" ($\rho = " + f"{self._correlation:.3f}$)")


class VIFPlotter(Plotter):
    """Plotter for Variance Inflation Factor (VIF)."""

    def __init__(
        self,
        vif: pd.DataFrame | np.ndarray,
        predictor_names: list[str] | None = None,
        config: MACConfig | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(config)
        if isinstance(vif, np.ndarray):
            if predictor_names is None:
                predictor_names = [f"x{i}" for i in range(vif.shape[0])]
            vif = pd.Series(vif, index=predictor_names, name="VIF").to_frame()
        if "VIF" not in vif.columns:
            raise ValueError("VIF DataFrame must contain a 'VIF' column.")
        self._vif = vif

    def _determine_figure_size(self, config: MACConfig) -> tuple[float, float]:
        """Determine figure size based on number of predictors."""
        n_predictors = self._vif.shape[0]
        height = n_predictors * config.bar_height + config.plot_bloat_height
        return (config.base_width, height)

    def _plot(self, ax: plt.Axes, masks: ProblematicSampleMasks | None = None) -> None:
        """Internal method to plot the VIF on the given Axes object."""

        cmap = LinearSegmentedColormap.from_list(
            "vif_cmap", [self._config.colors.marker_edge, self._config.colors.outlier_edge]
        )
        values = self._vif["VIF"].values
        colors = _compute_color_values(
            values, cmap, self._config.vif_lower_bound, self._config.vif_threshold
        )

        bar = ax.barh(self._vif.index, values, color=colors, alpha=1)
        annotations = ax.bar_label(bar, fmt="%.1f", label_type="edge", padding=3)
        ax.axvline(
            self._config.vif_threshold, color=self._config.colors.outlier_edge, linestyle="--"
        )
        ax.set_xlabel("Variance Inflation Factor (VIF)")
        for ann in annotations:
            bbox = ann.get_window_extent().transformed(ax.transData.inverted())
            ax.update_datalim(bbox.corners())
        ax.autoscale_view()
        ax.set_ylim(-0.5, len(values) - 0.5)


def _apply_lowess(x: np.ndarray, y: np.ndarray, config: MACConfig) -> np.ndarray:
    """Apply LOWESS smoothing to the data."""
    diff_x = Range(x).diff()
    delta = config.lowess_rel_delta * diff_x
    lowess_result = lowess(y, x, delta=delta, frac=config.lowess_frac, return_sorted=True)
    return lowess_result
