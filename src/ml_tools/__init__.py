"""Tools for machine learning."""

from ml_tools.mac.analyze import MAC, ModelAdequacyChecker
from ml_tools.mac.config import MACConfig
from ml_tools.mac.detection import ProblematicSampleMasks
from ml_tools.mac.fit import FitSummary
from ml_tools.mac.metric import MetricSummary

from . import datasets

__all__ = [
    "datasets",
    "MAC",
    "ModelAdequacyChecker",
    "MACConfig",
    "ProblematicSampleMasks",
    "FitSummary",
    "MetricSummary",
]
