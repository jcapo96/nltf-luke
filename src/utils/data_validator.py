"""
Utility class for data validation operations.

This module provides validation utilities for file paths, data types,
and data integrity checks used throughout the framework.
"""

import pandas as pd
from typing import Any


class DataValidator:
    """
    Utility class for data validation operations.

    This class provides static methods for validating various aspects
    of data and file paths used in the framework.
    """

    @staticmethod
    def validate_file_path(path: str) -> str:
        """
        Validate and return the dataset type from file path.

        Args:
            path: File path to validate

        Returns:
            Dataset type string (Baseline, Ullage, Liquid, or Unknown)

        Raises:
            ValueError: If path is invalid or doesn't point to Excel file
        """
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string.")
        if not path.endswith('.xlsx'):
            raise ValueError("Path must point to an Excel file with .xlsx extension.")

        if "baseline" in path:
            return "Baseline"
        elif "ullage" in path:
            return "Ullage"
        elif "liquid" in path:
            return "Liquid"
        else:
            return "Unknown"

    @staticmethod
    def validate_dataframe(data: Any, name: str) -> None:
        """
        Validate that data is a non-empty pandas DataFrame.

        Args:
            data: Data to validate
            name: Name of the data for error messages

        Raises:
            TypeError: If data is not a pandas DataFrame
            ValueError: If DataFrame is empty
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        if data.empty:
            raise ValueError(f"{name} cannot be empty.")
