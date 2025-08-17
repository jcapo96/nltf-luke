"""
Standard data format that the framework expects.

This format is independent of the original data source and provides
a consistent interface for all data processors.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class StandardDataFormat:
    """
    Standard data format that the framework expects.

    This format is independent of the original data source and provides
    a consistent interface for all data processors.
    """

    # Core data columns with standardized names
    timestamp: pd.Series  # Primary timestamp column
    liquid_level: Optional[pd.Series] = None  # Liquid level measurements
    h2o_concentration: Optional[pd.Series] = None  # H2O concentration measurements
    temperature: Optional[pd.Series] = None  # Temperature measurements
    purity: Optional[pd.Series] = None  # Purity/lifetime measurements

    # Metadata
    dataset_name: str = ""
    source_file: str = ""

    # Additional timestamp columns for different signals (if multiple exist)
    additional_timestamps: Dict[str, pd.Series] = None

    def __post_init__(self):
        """Validate the standard format after initialization."""
        if self.timestamp is None or self.timestamp.empty:
            raise ValueError("Timestamp column is required and cannot be empty")

        # Ensure all series have the same length as timestamp
        # But allow signals to have their own lengths since they may have different sampling rates
        for field_name, field_value in self.__dict__.items():
            if (isinstance(field_value, pd.Series) and
                field_name != 'timestamp' and
                field_value is not None):
                # Don't enforce length matching - each signal can have its own length
                # Just ensure the series is valid
                if field_value.empty:
                    print(f"Warning: Column {field_name} is empty")

        # Initialize additional_timestamps if None
        if self.additional_timestamps is None:
            self.additional_timestamps = {}
