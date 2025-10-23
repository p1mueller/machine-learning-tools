"""Tools for machine learning."""

from ml_tools.mac import analyze, config, detection, fit, metric, plot
from ml_tools.mac.analyze import ModelAdequacyChecker
from ml_tools.mac.config import MACConfig
from ml_tools.mac.detection import ProblematicSampleMasks
from ml_tools.mac.fit import FitSummary
from ml_tools.mac.metric import MetricSummary

from . import datasets

__all__ = [
    "datasets",
    "ModelAdequacyChecker",
    "MACConfig",
    "ProblematicSampleMasks",
    "FitSummary",
    "MetricSummary",
    "analyze",
    "config",
    "detection",
    "fit",
    "metric",
    "plot",
]
