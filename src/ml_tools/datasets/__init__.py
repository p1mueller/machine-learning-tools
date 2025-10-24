"""Dataset module for ml_tools package."""

from ml_tools.datasets.advertising import AdvertisingDataset
from ml_tools.datasets.auto import AutoDataset
from ml_tools.datasets.injection_molding import InjectionMoldingDataset

__all__ = [
    "AdvertisingDataset",
    "AutoDataset",
    "InjectionMoldingDataset",
]
