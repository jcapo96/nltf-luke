"""
Data Format Abstraction Layer

This module provides a standard interface for data formats and converters
to transform different input formats into the standard format expected by the framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import os


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


class BaseDataConverter(ABC):
    """Abstract base class for data converters."""

    @abstractmethod
    def can_convert(self, file_path: str) -> bool:
        """Check if this converter can handle the given file."""
        pass

    @abstractmethod
    def convert(self, file_path: str) -> StandardDataFormat:
        """Convert the input file to standard format."""
        pass

    @abstractmethod
    def get_dataset_type(self, file_path: str) -> str:
        """Extract dataset type from file path."""
        pass


class SeeqNewConverter(BaseDataConverter):
    """Converts Excel files from Seeq new format to StandardDataFormat."""

    def can_convert(self, file_path: str) -> bool:
        """Check if this converter can handle the given file."""
        return file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls')

    def get_dataset_type(self, file_path: str) -> str:
        """Extract dataset type from filename."""
        filename = os.path.basename(file_path).lower()
        if 'baseline' in filename:
            return 'baseline'
        elif 'ullage' in filename:
            return 'ullage'
        elif 'liquid' in filename:
            return 'liquid'
        else:
            return 'unknown'

    def convert(self, file_path: str) -> StandardDataFormat:
        """Convert Excel file to standard format."""
        if not self.can_convert(file_path):
            raise ValueError(f"Cannot convert file {file_path} - format not supported")

        # Read the data sheet
        data_df = pd.read_excel(file_path, sheet_name='Data')

        # Extract dataset type from filename
        dataset_type = self.get_dataset_type(file_path)

        # Find timestamp columns and signal columns
        timestamp_cols = [col for col in data_df.columns if 'Date-Time' in col]
        if not timestamp_cols:
            raise ValueError("No timestamp columns found in data sheet")

        primary_timestamp = timestamp_cols[0]

        # Convert timestamp to datetime
        data_df[primary_timestamp] = pd.to_datetime(data_df[primary_timestamp], errors='coerce')

        # Don't drop rows based on primary timestamp - each signal has its own timestamp
        # data_df = data_df.dropna(subset=[primary_timestamp])

        # if data_df.empty:
        #     raise ValueError("No valid timestamps found in data")

        # Process each signal independently with its own timestamp

        # Extract signal columns
        liquid_level = None
        h2o_concentration = None
        temperature = None
        purity = None
        additional_timestamps = {}

        # Define the signal columns we're looking for
        signal_columns = {
            'PAB_S1_LT_13_AR_REAL_F_CV': 'liquid_level',
            'PAB_S1_AE_611_AR_REAL_F_CV': 'h2o_concentration',
            'Luke_PRM_LIFETIME_F_CV': 'purity',
            'PAB_S1_TE_324_AR_REAL_F_CV': 'temperature'
        }

        # Get all column names
        all_columns = data_df.columns.tolist()

        for signal_col, signal_name in signal_columns.items():
            if signal_col in all_columns:
                # Find the index of the signal column
                signal_idx = all_columns.index(signal_col)

                # The timestamp column should be immediately to the left (index - 1)
                if signal_idx > 0:
                    timestamp_col = all_columns[signal_idx - 1]

                    # Verify it's a timestamp column (contains 'Date-Time')
                    if 'Date-Time' in timestamp_col:
                        # Get the signal data
                        signal_data = data_df[signal_col]

                        # Get the corresponding timestamp
                        signal_timestamp = pd.to_datetime(data_df[timestamp_col], errors='coerce')

                        # Create a series with the signal's own timestamp
                        signal_series = pd.Series(signal_data.values, index=signal_timestamp)

                        # Drop rows with invalid timestamps
                        signal_series = signal_series.dropna()

                        # Assign to appropriate field
                        if signal_name == 'liquid_level':
                            liquid_level = signal_series
                        elif signal_name == 'h2o_concentration':
                            h2o_concentration = signal_series
                        elif signal_name == 'temperature':
                            temperature = signal_series
                        elif signal_name == 'purity':
                            purity = signal_series

                        print(f"Found {signal_name}: {signal_col} with timestamp {timestamp_col} ({len(signal_series)} points)")
                    else:
                        print(f"Warning: Expected timestamp column before {signal_col}, but found {timestamp_col}")
                else:
                    print(f"Warning: Signal column {signal_col} is the first column, no timestamp column found")
            else:
                print(f"Warning: Signal column {signal_col} not found in data")

        # For the primary timestamp, use only the non-null values
        # This will be used as a reference timestamp for the overall dataset
        primary_timestamp_clean = pd.to_datetime(data_df[primary_timestamp], errors='coerce').dropna()

        # Create standard format
        # Allow each signal to maintain its natural length and index
        return StandardDataFormat(
            timestamp=primary_timestamp_clean,
            liquid_level=liquid_level,
            h2o_concentration=h2o_concentration,
            temperature=temperature,
            purity=purity,
            dataset_name=dataset_type,
            source_file=file_path,
            additional_timestamps=additional_timestamps
        )


class SeeqOldConverter(BaseDataConverter):
    """Converts Excel files from Seeq old format to StandardDataFormat."""

    def can_convert(self, file_path: str) -> bool:
        """Check if this converter can handle the given file."""
        return file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls')

    def get_dataset_type(self, file_path: str) -> str:
        """Extract dataset type from filename."""
        filename = os.path.basename(file_path).lower()
        if 'baseline' in filename:
            return 'baseline'
        elif 'ullage' in filename:
            return 'ullage'
        elif 'liquid' in filename:
            return 'liquid'
        else:
            return 'unknown'

    def convert(self, file_path: str) -> StandardDataFormat:
        """Convert the Excel file to StandardDataFormat."""
        if not self.can_convert(file_path):
            raise ValueError(f"Cannot convert file {file_path} - format not supported")

        # Read the data sheet
        data_df = pd.read_excel(file_path, sheet_name='Data')

        # Extract dataset type from filename
        dataset_type = self.get_dataset_type(file_path)

        # Find timestamp columns and signal columns
        timestamp_cols = [col for col in data_df.columns if 'Date-Time' in col]
        if not timestamp_cols:
            raise ValueError("No timestamp columns found in data sheet")

        primary_timestamp = timestamp_cols[0]

        # Convert timestamp to datetime
        data_df[primary_timestamp] = pd.to_datetime(data_df[primary_timestamp], errors='coerce')

        # Don't drop rows based on primary timestamp - each signal has its own timestamp
        # data_df = data_df.dropna(subset=[primary_timestamp])

        # if data_df.empty:
        #     raise ValueError("No valid timestamps found in data")

        # Process each signal independently with its own timestamp

        # Extract signal columns
        liquid_level = None
        h2o_concentration = None
        temperature = None
        purity = None
        additional_timestamps = {}

        # Define the signal columns we're looking for (different H2O column from SeeqNewConverter)
        signal_columns = {
            'PAB_S1_LT_13_AR_REAL_F_CV': 'liquid_level',
            'PAB_S1.AE_600_AR_REAL.F_CV': 'h2o_concentration',  # Different H2O column
            'Luke_PRM_LIFETIME_F_CV': 'purity',
            'PAB_S1_TE_324_AR_REAL_F_CV': 'temperature'
        }

        # Get all column names
        all_columns = data_df.columns.tolist()

        for signal_col, signal_name in signal_columns.items():
            if signal_col in all_columns:
                # Find the index of the signal column
                signal_idx = all_columns.index(signal_col)

                # The timestamp column should be immediately to the left (index - 1)
                if signal_idx > 0:
                    timestamp_col = all_columns[signal_idx - 1]

                    # Verify it's a timestamp column (contains 'Date-Time')
                    if 'Date-Time' in timestamp_col:
                        # Get the signal data
                        signal_data = data_df[signal_col]

                        # Get the corresponding timestamp
                        signal_timestamp = pd.to_datetime(data_df[timestamp_col], errors='coerce')

                        # Create a series with the signal's own timestamp
                        signal_series = pd.Series(signal_data.values, index=signal_timestamp)

                        # Drop rows with invalid timestamps
                        signal_series = signal_series.dropna()

                        # Assign to appropriate field
                        if signal_name == 'liquid_level':
                            liquid_level = signal_series
                        elif signal_name == 'h2o_concentration':
                            h2o_concentration = signal_series
                        elif signal_name == 'temperature':
                            temperature = signal_series
                        elif signal_name == 'purity':
                            purity = signal_series

                        print(f"Found {signal_name}: {signal_col} with timestamp {timestamp_col} ({len(signal_series)} points)")
                    else:
                        print(f"Warning: Expected timestamp column before {signal_col}, but found {timestamp_col}")
                else:
                    print(f"Warning: Signal column {signal_col} is the first column, no timestamp column found")
            else:
                print(f"Warning: Signal column {signal_col} not found in data")

        # For the primary timestamp, use only the non-null values
        # This will be used as a reference timestamp for the overall dataset
        primary_timestamp_clean = pd.to_datetime(data_df[primary_timestamp], errors='coerce').dropna()

        # Create standard format
        # Allow each signal to maintain its natural length and index
        return StandardDataFormat(
            timestamp=primary_timestamp_clean,
            liquid_level=liquid_level,
            h2o_concentration=h2o_concentration,
            temperature=temperature,
            purity=purity,
            dataset_name=dataset_type,
            source_file=file_path,
            additional_timestamps=additional_timestamps
        )

    def _align_signal_with_timestamp(self, signal_timestamp: pd.Series, signal_values: pd.Series,
                                   primary_timestamp: pd.Series) -> pd.Series:
        """
        Align signal data with primary timestamp using interpolation.

        This handles cases where different signals have their own timestamp columns.
        """
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(signal_timestamp):
            signal_timestamp = pd.to_datetime(signal_timestamp, errors='coerce')

        # Create a temporary dataframe for interpolation
        temp_df = pd.DataFrame({
            'timestamp': signal_timestamp,
            'values': signal_values
        }).dropna()

        if temp_df.empty:
            return pd.Series(index=primary_timestamp, dtype=float)

        # Sort by timestamp
        temp_df = temp_df.sort_values('timestamp').reset_index(drop=True)

        # Interpolate to primary timestamp
        aligned_values = np.interp(
            primary_timestamp.astype(np.int64),
            temp_df['timestamp'].astype(np.int64),
            temp_df['values']
        )

        return pd.Series(aligned_values, index=primary_timestamp)

    def _ensure_consistent_index(self, data_df: pd.DataFrame, primary_timestamp: pd.Series) -> pd.DataFrame:
        """
        Ensure all data series have the same index structure.
        """
        # Reset index to match the primary timestamp
        data_df = data_df.reset_index(drop=True)

        # Create a new dataframe with consistent indexing
        aligned_df = pd.DataFrame(index=range(len(primary_timestamp)))
        aligned_df['timestamp'] = primary_timestamp

        # Align all signal columns
        for col in data_df.columns:
            if col != primary_timestamp.name and 'Date-Time' not in col:
                if col in data_df.columns:
                    aligned_df[col] = data_df[col].values

        return aligned_df

    def get_dataset_type(self, file_path: str) -> str:
        """Extract dataset type from file path."""
        filename = file_path.lower()
        if "baseline" in filename:
            return "baseline"
        elif "ullage" in filename:
            return "ullage"
        elif "liquid" in filename:
            return "liquid"
        else:
            return "unknown"


class DataFormatManager:
    """
    Manages data format conversion and provides a unified interface.
    """

    def __init__(self):
        self.converters: List[BaseDataConverter] = []
        self._register_default_converters()

    def _register_default_converters(self):
        """Register the default converters."""
        self.register_converter(SeeqNewConverter())
        self.register_converter(SeeqOldConverter())

    def register_converter(self, converter: BaseDataConverter):
        """Register a new data converter."""
        self.converters.append(converter)

    def convert_file(self, file_path: str) -> StandardDataFormat:
        """Convert a file to standard format using the appropriate converter."""
        for converter in self.converters:
            if converter.can_convert(file_path):
                return converter.convert(file_path)

        # If no converter found, try to provide helpful error message
        supported_formats = [f"{converter.__class__.__name__}" for converter in self.converters]
        raise ValueError(
            f"No converter found for file {file_path}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    def can_convert(self, file_path: str) -> bool:
        """Check if any converter can handle the given file."""
        for converter in self.converters:
            if converter.can_convert(file_path):
                return True
        return False

    def get_converter(self, file_path: str) -> BaseDataConverter:
        """Get the converter that can handle the given file."""
        for converter in self.converters:
            if converter.can_convert(file_path):
                return converter
        raise ValueError(f"No converter found for file {file_path}")

    def get_supported_formats(self) -> List[str]:
        """Get list of supported format names."""
        return [converter.__class__.__name__ for converter in self.converters]
