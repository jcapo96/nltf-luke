"""
Dataset module for managing data and datasets.

This module contains the Dataset class for handling individual datasets
and the DatasetManager class for coordinating multiple datasets.
"""

from .dataset import Dataset
from .dataset_manager import DatasetManager

__all__ = [
    'Dataset',
    'DatasetManager'
]
