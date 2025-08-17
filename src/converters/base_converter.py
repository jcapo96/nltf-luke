"""
Abstract base class for data converters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import os
from core.standard_format import StandardDataFormat


class BaseDataConverter(ABC):
    """
    Abstract base class for data converters.

    To implement a new converter:
    1. Inherit from this class
    2. Implement the three required methods: can_convert, convert, get_dataset_type
    3. Register your converter with DataFormatManager
    """

    @abstractmethod
    def can_convert(self, file_path: str) -> bool:
        """
        Check if this converter can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this converter can handle the file, False otherwise

        Example:
            return file_path.lower().endswith('.csv')  # For CSV files
        """
        pass

    @abstractmethod
    def convert(self, file_path: str) -> StandardDataFormat:
        """
        Convert the input file to standard format.

        Args:
            file_path: Path to the file to convert

        Returns:
            StandardDataFormat object with converted data

        This is the main method where you implement your conversion logic.
        You should:
        1. Read the input file
        2. Extract the required signals (liquid_level, h2o_concentration, temperature, purity)
        3. Extract timestamps for each signal
        4. Return a StandardDataFormat object
        """
        pass

    @abstractmethod
    def get_dataset_type(self, file_path: str) -> str:
        """
        Extract dataset type from file path.

        Args:
            file_path: Path to the file

        Returns:
            String indicating dataset type: 'baseline', 'ullage', 'liquid', or 'unknown'

        This method helps identify which type of dataset the file contains.
        """
        pass
