"""Utility functions."""

from pathlib import Path


def get_data_folder():
    """Returns the path to the data folder."""
    return Path(__file__).parents[2] / "data"
