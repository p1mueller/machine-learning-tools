"""Tools for machine learning."""

from ml_tools.mac.analyze import MAC, ModelAdequacyChecker
from ml_tools.mac.config import MACConfig
from ml_tools.mac.detection import ProblematicSampleMasks
from ml_tools.mac.fit import FitSummary
from ml_tools.mac.metric import MetricSummary
from ml_tools.mac.report import (
    FigureImage,
    HTMLReport,
    MACReport,
    MarkdownReport,
    PredictorReport,
    ProblemSummary,
    TextReport,
    render_plotters,
)

from . import datasets

__all__ = [
    "datasets",
    "MAC",
    "ModelAdequacyChecker",
    "MACConfig",
    "ProblematicSampleMasks",
    "FitSummary",
    "MetricSummary",
    "MACReport",
    "TextReport",
    "MarkdownReport",
    "HTMLReport",
    "PredictorReport",
    "ProblemSummary",
    "FigureImage",
    "render_plotters",
]
