"""Utility functions."""

from pathlib import Path


def get_data_folder() -> Path:
    """Returns the path to the package data folder."""
    return Path(__file__).parent / "data"
