"""Core report model and shared helpers for MAC reports."""

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ml_tools.mac.analyze import DiagnosticPlots
from ml_tools.mac.config import MACConfig, get_default_config
from ml_tools.mac.detection import ProblematicSampleMasks
from ml_tools.mac.metric import MetricSummary
from ml_tools.mac.plot import Plotter


class PredictorReport(BaseModel):
    """Report entry for a single predictor variable."""

    name: str = Field(description="Name of the predictor")
    vif: float = Field(description="Variance Inflation Factor of the predictor")
    flagged: bool = Field(description="Whether the VIF exceeds the configured threshold")
    p_value: float | None = Field(
        default=None, description="P-value for testing the coefficient against zero"
    )


class ProblemSummary(BaseModel):
    """Indices of problematic samples, grouped by category."""

    outliers: list[int] = Field(default_factory=list, description="Indices of outlier samples")
    high_leverage: list[int] = Field(
        default_factory=list, description="Indices of high leverage samples"
    )
    influential: list[int] = Field(
        default_factory=list, description="Indices of influential samples"
    )

    def items(self) -> list[tuple[str, list[int]]]:
        """Categories and their indices, in display order."""
        return [
            ("Outliers", self.outliers),
            ("High leverage", self.high_leverage),
            ("Influential", self.influential),
        ]

    @property
    def total(self) -> int:
        """Total number of flagged sample indices across all categories."""
        return len(self.outliers) + len(self.high_leverage) + len(self.influential)


class FigureImage(BaseModel):
    """A rendered diagnostic plot, embedded as a base64 PNG data URI."""

    title: str = Field(description="Title of the figure")
    image: str = Field(description="Png image encoded as a data URI (data:image/png;base64,...)")


def fmt(value: float | None) -> str:
    """Format a numeric value, mapping NaN to ``n/a`` and None to ``n/a``."""
    if value is None:
        return "n/a"
    try:
        if np.isnan(value):
            return "n/a"
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, float) and float(value).is_integer()
    ):
        return str(int(value))
    return f"{value:.4g}"


def _optional_float(value: float) -> float | None:
    """Convert an NaN (or non-finite) float to None for optional fields."""
    try:
        if not np.isfinite(value):
            return None
    except TypeError:
        return None
    return float(value)


def vif_status(vif: float, threshold: float) -> str:
    """Markdown status text for a VIF value against a threshold."""
    return f":warning: flagged (>{threshold:g})" if vif > threshold else "ok"


def format_indices(indices: list[int], max_indices: int) -> str:
    """Format a list of indices, truncating it to ``max_indices`` entries."""
    if not indices:
        return "none"
    shown = indices[:max_indices]
    extra = len(indices) - max_indices
    suffix = f", ... (+{extra} more)" if extra > 0 else ""
    return ", ".join(str(i) for i in shown) + suffix


class MACReport(BaseModel):
    """Base report of a model adequacy analysis.

    Holds the model statistics, per-predictor VIF diagnostics and the
    indices of problematic samples. It is data-only: rendering is delegated
    to the adapter subclasses [`TextReport`][...TextReport],
    [`MarkdownReport`][...MarkdownReport] and
    [`HTMLReport`][...HTMLReport], each implementing
    [`render`][.render].
    """

    title: str = Field(default="Model Adequacy Report", description="Title of the report")
    n_samples: int = Field(description="Number of samples")
    n_features: int = Field(description="Number of predictor variables")
    has_intercept: bool = Field(description="Whether the model includes an intercept")
    dof: int = Field(description="Number of fitted parameters (degrees of freedom)")
    r_squared: float = Field(description="Coefficient of determination")
    adj_r_squared: float = Field(description="Adjusted coefficient of determination")
    rse: float = Field(description="Residual standard error")
    rss: float = Field(description="Residual sum of squares")
    tss: float = Field(description="Total sum of squares")
    f_statistic: float = Field(description="Overall F-statistic of the model")
    model_p_value: float = Field(
        default=float("nan"),
        description="P-value of the overall model F-test (all slopes zero)",
    )
    residual_correlation: float = Field(description="First-lag autocorrelation of the residuals")
    vif_threshold: float = Field(description="VIF threshold used to flag predictors")
    predictors: list[PredictorReport] = Field(description="Per-predictor VIF diagnostics")
    problems: ProblemSummary = Field(
        description="Indices of problematic samples", default_factory=ProblemSummary
    )
    figures: list[FigureImage] = Field(
        default_factory=list,
        description="Diagnostic plots rendered to base64 PNGs (see HTMLReport)",
    )

    @classmethod
    def from_analysis(
        cls,
        metric: MetricSummary,
        masks: ProblematicSampleMasks,
        plots: DiagnosticPlots,
        config: MACConfig | None = None,
    ) -> Self:
        """Build a report directly from the results of a ``ModelAdequacyChecker`` analysis.

        Args:
            metric: The metric summary returned by the analysis.
            masks: The problematic sample masks returned by the analysis.
            plots: The diagnostic plotters returned by the analysis; they are
                rendered into inline base64 PNGs for the HTML report.
            config: Optional configuration used for thresholds. If None, the
                default configuration is used.

        Examples:
            >>> checker = ModelAdequacyChecker()  # doctest: +SKIP
            >>> metric, masks, plots = checker.analyze(summary, plot=False)  # doctest: +SKIP
            >>> report = HTMLReport.from_analysis(metric, masks, plots)  # doctest: +SKIP
            >>> report.save("report.html")  # doctest: +SKIP
        """

        if config is None:
            config = get_default_config()
        summary = metric.summary
        problems = ProblemSummary(
            outliers=np.flatnonzero(masks.outliers).tolist(),
            high_leverage=np.flatnonzero(masks.high_leverage).tolist(),
            influential=np.flatnonzero(masks.influential).tolist(),
        )
        vif_df = metric.pretty_vif(summary.predictor_names)
        coef_p = metric.coefficient_p_values
        p_offset = 1 if summary.has_intercept else 0
        predictors = [
            PredictorReport(
                name=name,
                vif=float(vif),
                flagged=float(vif) > config.vif_threshold,
                p_value=_optional_float(coef_p[i + p_offset]),
            )
            for i, (name, vif) in enumerate(zip(vif_df.index, vif_df["VIF"]))
        ]
        figures = render_plotters(plots, masks=masks)
        return cls(
            n_samples=metric.n_samples,
            n_features=len(summary.predictor_names),
            has_intercept=summary.has_intercept,
            dof=metric.dof,
            r_squared=metric.r_squared,
            adj_r_squared=metric.adj_r_squared,
            rse=metric.rse,
            rss=metric.rss,
            tss=metric.tss,
            f_statistic=metric.f_statistic,
            model_p_value=metric.model_p_value,
            residual_correlation=metric.residual_correlation,
            vif_threshold=config.vif_threshold,
            predictors=predictors,
            problems=problems,
            figures=figures,
        )

    def add_figures(self, figures: list[FigureImage]) -> Self:
        """Return a copy of the report with the given diagnostic figures attached."""
        return self.model_copy(update={"figures": list(self.figures) + list(figures)})

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain nested dictionary."""
        return self.model_dump(mode="json")

    def to_dataframe(self) -> pd.DataFrame:
        """Return the model statistics as a two-column DataFrame."""
        stats = {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "has_intercept": self.has_intercept,
            "dof": self.dof,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "rse": self.rse,
            "rss": self.rss,
            "tss": self.tss,
            "f_statistic": self.f_statistic,
            "model_p_value": self.model_p_value,
            "residual_correlation": self.residual_correlation,
        }
        return pd.Series(stats).to_frame().T

    def to_df_vif(self) -> pd.DataFrame:
        """Return the per-predictor VIF table."""
        return pd.DataFrame(
            {
                "VIF": [p.vif for p in self.predictors],
                "flagged": [p.flagged for p in self.predictors],
            },
            index=pd.Index([p.name for p in self.predictors], name="name"),
        )

    def render(self, max_indices: int = 20) -> str:
        """Render the report as a string.

        The base class is data-only and does not render; use one of the
        adapter subclasses instead.

        Args:
            max_indices: Maximum number of sample indices listed per category.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not render; use an adapter such as "
            "TextReport, MarkdownReport or HTMLReport."
        )

    def save(
        self,
        path: str | os.PathLike[str],
        max_indices: int = 20,
    ) -> None:
        """Render the report and write it to a file.

        Args:
            path: Destination file path.
            max_indices: Maximum number of sample indices listed per category.
        """
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render(max_indices=max_indices))

    def _stat_pairs(self) -> list[tuple[str, float]]:
        """Model statistics as ``(label, value)`` pairs in display order."""
        return [
            ("Number of samples", float(self.n_samples)),
            ("Number of features", float(self.n_features)),
            ("Intercept", float(self.has_intercept)),
            ("Degrees of freedom", float(self.dof)),
            ("R²", self.r_squared),
            ("Adjusted R²", self.adj_r_squared),
            ("RSE", self.rse),
            ("RSS", self.rss),
            ("TSS", self.tss),
            ("F-statistic", self.f_statistic),
            ("Model p-value", self.model_p_value),
            ("Residual correlation (lag 1)", self.residual_correlation),
        ]

    def __str__(self) -> str:  # noqa: D105
        return self.render()


def render_plotters(
    plotters: DiagnosticPlots | dict[str, Plotter],
    masks: ProblematicSampleMasks | None = None,
    dpi: int = 150,
    titles: dict[str, str] | None = None,
) -> list[FigureImage]:
    """Render plotters into base64-encoded [`FigureImage`][..FigureImage] objects.

    Figures created by this call are closed afterwards. Plotters that were
    already rendered (see [`Plotter.figure`][ml_tools.mac.plot.Plotter.figure] are
    reused and left open, so previously shown figures stay usable.

    Args:
        plotters: A [`DiagnosticPlots`][ml_tools.mac.analyze.DiagnosticPlots] object or
            a mapping of name to plotter.
        masks: Optional [`ProblematicSampleMasks`][ml_tools.mac.detection.ProblematicSampleMasks]
            to highlight problematic samples in the plots.
        dpi: Resolution used when saving the figures.
        titles: Optional override of figure titles, keyed by plot name.
    """

    if isinstance(plotters, DiagnosticPlots):
        items = plotters.model_dump().items()
    else:
        items = dict(plotters).items()
    figures: list[FigureImage] = []
    for name, plotter in items:
        title = (titles or {}).get(name) or getattr(plotter, "title", None) or name
        render_now = plotter.figure is None
        fig, _ax = plotter.plot(masks=masks)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        if render_now:
            plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        figures.append(FigureImage(title=title, image=f"data:image/png;base64,{encoded}"))
    return figures
