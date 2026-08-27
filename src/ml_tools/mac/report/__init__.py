"""Report generation for model adequacy analysis."""

from .core import (
    FigureImage,
    MACReport,
    PredictorReport,
    ProblemSummary,
    fmt,
    format_indices,
    render_plotters,
    vif_status,
)
from .html import HTMLReport
from .markdown import MarkdownReport
from .text import TextReport

__all__ = [
    "FigureImage",
    "MACReport",
    "PredictorReport",
    "ProblemSummary",
    "TextReport",
    "MarkdownReport",
    "HTMLReport",
    "format_indices",
    "fmt",
    "render_plotters",
    "vif_status",
]
