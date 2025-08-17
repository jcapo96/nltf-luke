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


class SeeqNewConverter(BaseDataConverter):
    """
    Converts Excel files from Seeq new format to StandardDataFormat.

    This converter expects Excel files with a 'Data' sheet containing:
    - Multiple timestamp columns (containing 'Date-Time' in the name)
    - Signal columns with specific names (see signal_columns dictionary)
    - Each signal column is preceded by its corresponding timestamp column

    To modify for different signal names:
    1. Update the signal_columns dictionary below
    2. Ensure the timestamp column logic matches your data structure
    """

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
        """
        Convert Excel file to standard format.

        This method implements the conversion logic for Seeq new format:
        1. Reads the 'Data' sheet from Excel
        2. Identifies signal columns by their names
        3. Finds corresponding timestamp columns (immediately to the left of each signal)
        4. Creates pandas Series for each signal with their own timestamps
        5. Returns StandardDataFormat object
        """
        if not self.can_convert(file_path):
            raise ValueError(f"Cannot convert file {file_path} - format not supported")

        # Read the data sheet from Excel file
        data_df = pd.read_excel(file_path, sheet_name='Data')

        # Extract dataset type from filename for metadata
        dataset_type = self.get_dataset_type(file_path)

        # Find all timestamp columns (columns containing 'Date-Time')
        timestamp_cols = [col for col in data_df.columns if 'Date-Time' in col]
        if not timestamp_cols:
            raise ValueError("No timestamp columns found in data sheet")

        # Use the first timestamp column as primary reference
        primary_timestamp = timestamp_cols[0]

        # Convert primary timestamp to datetime format
        data_df[primary_timestamp] = pd.to_datetime(data_df[primary_timestamp], errors='coerce')

        # Initialize variables to store extracted signals
        liquid_level = None
        h2o_concentration = None
        temperature = None
        purity = None
        additional_timestamps = {}

        # Define the mapping between Excel column names and standard signal names
        # MODIFY THIS DICTIONARY to add/remove signals or change column names
        signal_columns = {
            'PAB_S1_LT_13_AR_REAL_F_CV': 'liquid_level',      # Liquid level sensor
            'PAB_S1_AE_611_AR_REAL_F_CV': 'h2o_concentration', # H2O concentration sensor
            'Luke_PRM_LIFETIME_F_CV': 'purity',                # Purity/lifetime sensor
            'PAB_S1_TE_324_AR_REAL_F_CV': 'temperature'       # Temperature sensor
        }

        # Get all column names from the Excel sheet
        all_columns = data_df.columns.tolist()

        # Process each signal column
        for signal_col, signal_name in signal_columns.items():
            if signal_col in all_columns:
                # Find the position of the signal column in the Excel sheet
                signal_idx = all_columns.index(signal_col)

                # The timestamp column should be immediately to the left (index - 1)
                # This assumes the Excel structure: [Timestamp1, Signal1, Timestamp2, Signal2, ...]
                if signal_idx > 0:
                    timestamp_col = all_columns[signal_idx - 1]

                    # Verify it's actually a timestamp column (contains 'Date-Time')
                    if 'Date-Time' in timestamp_col:
                        # Extract the signal data values
                        signal_data = data_df[signal_col]

                        # Extract the corresponding timestamp column and convert to datetime
                        signal_timestamp = pd.to_datetime(data_df[timestamp_col], errors='coerce')

                        # Create a pandas Series with the signal's own timestamp
                        # This allows each signal to maintain its natural length and sampling rate
                        signal_series = pd.Series(signal_data.values, index=signal_timestamp)

                        # Remove rows with invalid timestamps (NaN values)
                        signal_series = signal_series.dropna()

                        # Assign the processed signal to the appropriate field
                        if signal_name == 'liquid_level':
                            liquid_level = signal_series
                        elif signal_name == 'h2o_concentration':
                            h2o_concentration = signal_series
                        elif signal_name == 'temperature':
                            temperature = signal_series
                        elif signal_name == 'purity':
                            purity = signal_series

                        # Log successful signal extraction (useful for debugging)
                        print(f"Found {signal_name}: {signal_col} with timestamp {timestamp_col} ({len(signal_series)} points)")
                    else:
                        # Log warning if expected timestamp column is not found
                        print(f"Warning: Expected timestamp column before {signal_col}, but found {timestamp_col}")
                else:
                    # Log warning if signal column is the first column (no timestamp before it)
                    print(f"Warning: Signal column {signal_col} is the first column, no timestamp column found")
            else:
                # Log warning if expected signal column is not found in the Excel sheet
                print(f"Warning: Signal column {signal_col} not found in data")

        # Clean the primary timestamp column (remove NaN values)
        # This will be used as a reference timestamp for the overall dataset
        primary_timestamp_clean = pd.to_datetime(data_df[primary_timestamp], errors='coerce').dropna()

        # Create and return the StandardDataFormat object
        # Each signal maintains its natural length and timestamp index
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
    """
    Converts Excel files from Seeq old format to StandardDataFormat.

    This converter is identical to SeeqNewConverter except for the H2O column name.
    It handles legacy data where H2O concentration uses a different column identifier.

    To modify for different legacy formats:
    1. Update the signal_columns dictionary below
    2. Ensure the timestamp column logic matches your data structure
    """

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
        """
        Convert the Excel file to StandardDataFormat.

        This method implements the conversion logic for Seeq old format.
        The main difference from SeeqNewConverter is the H2O column name.
        """
        if not self.can_convert(file_path):
            raise ValueError(f"Cannot convert file {file_path} - format not supported")

        # Read the data sheet from Excel file
        data_df = pd.read_excel(file_path, sheet_name='Data')

        # Extract dataset type from filename for metadata
        dataset_type = self.get_dataset_type(file_path)

        # Find all timestamp columns (columns containing 'Date-Time')
        timestamp_cols = [col for col in data_df.columns if 'Date-Time' in col]
        if not timestamp_cols:
            raise ValueError("No timestamp columns found in data sheet")

        # Use the first timestamp column as primary reference
        primary_timestamp = timestamp_cols[0]

        # Convert primary timestamp to datetime format
        data_df[primary_timestamp] = pd.to_datetime(data_df[primary_timestamp], errors='coerce')

        # Initialize variables to store extracted signals
        liquid_level = None
        h2o_concentration = None
        temperature = None
        purity = None
        additional_timestamps = {}

        # Define the mapping between Excel column names and standard signal names
        # NOTE: This is the ONLY difference from SeeqNewConverter - the H2O column name
        signal_columns = {
            'PAB_S1_LT_13_AR_REAL_F_CV': 'liquid_level',      # Liquid level sensor
            'PAB_S1.AE_600_AR_REAL.F_CV': 'h2o_concentration', # H2O concentration sensor (LEGACY NAME)
            'Luke_PRM_LIFETIME_F_CV': 'purity',                # Purity/lifetime sensor
            'PAB_S1_TE_324_AR_REAL_F_CV': 'temperature'       # Temperature sensor
        }

        # Get all column names from the Excel sheet
        all_columns = data_df.columns.tolist()

        # Process each signal column (same logic as SeeqNewConverter)
        for signal_col, signal_name in signal_columns.items():
            if signal_col in all_columns:
                # Find the position of the signal column in the Excel sheet
                signal_idx = all_columns.index(signal_col)

                # The timestamp column should be immediately to the left (index - 1)
                if signal_idx > 0:
                    timestamp_col = all_columns[signal_idx - 1]

                    # Verify it's actually a timestamp column (contains 'Date-Time')
                    if 'Date-Time' in timestamp_col:
                        # Extract the signal data values
                        signal_data = data_df[signal_col]

                        # Extract the corresponding timestamp column and convert to datetime
                        signal_timestamp = pd.to_datetime(data_df[timestamp_col], errors='coerce')

                        # Create a pandas Series with the signal's own timestamp
                        signal_series = pd.Series(signal_data.values, index=signal_timestamp)

                        # Remove rows with invalid timestamps (NaN values)
                        signal_series = signal_series.dropna()

                        # Assign the processed signal to the appropriate field
                        if signal_name == 'liquid_level':
                            liquid_level = signal_series
                        elif signal_name == 'h2o_concentration':
                            h2o_concentration = signal_series
                        elif signal_name == 'temperature':
                            temperature = signal_series
                        elif signal_name == 'purity':
                            purity = signal_series

                        # Log successful signal extraction
                        print(f"Found {signal_name}: {signal_col} with timestamp {timestamp_col} ({len(signal_series)} points)")
                    else:
                        # Log warning if expected timestamp column is not found
                        print(f"Warning: Expected timestamp column before {signal_col}, but found {timestamp_col}")
                else:
                    # Log warning if signal column is the first column (no timestamp before it)
                    print(f"Warning: Signal column {signal_col} is the first column, no timestamp column found")
            else:
                # Log warning if expected signal column is not found in the Excel sheet
                print(f"Warning: Signal column {signal_col} not found in data")

        # Clean the primary timestamp column (remove NaN values)
        primary_timestamp_clean = pd.to_datetime(data_df[primary_timestamp], errors='coerce').dropna()

        # Create and return the StandardDataFormat object
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


class DataFormatManager:
    """
    Manages data format conversion and provides a unified interface.

    This class acts as a central registry for all available converters.
    It automatically selects the appropriate converter for each file type.

    To add a new converter:
    1. Create a class that inherits from BaseDataConverter
    2. Implement the required methods
    3. Register it with this manager using register_converter()
    """

    def __init__(self):
        """Initialize the manager and register default converters."""
        self.converters: List[BaseDataConverter] = []
        self._register_default_converters()

    def _register_default_converters(self):
        """Register the default converters that come with the framework."""
        self.register_converter(SeeqNewConverter())
        self.register_converter(SeeqOldConverter())

    def register_converter(self, converter: BaseDataConverter):
        """
        Register a new data converter.

        Args:
            converter: Instance of a class that inherits from BaseDataConverter

        This method allows you to add custom converters to the framework.
        Converters are tried in the order they are registered.
        """
        self.converters.append(converter)

    def convert_file(self, file_path: str) -> StandardDataFormat:
        """
        Convert a file to standard format using the appropriate converter.

        Args:
            file_path: Path to the file to convert

        Returns:
            StandardDataFormat object with converted data

        This method automatically selects the first converter that can handle the file.
        If no converter is found, it raises a helpful error message.
        """
        for converter in self.converters:
            if converter.can_convert(file_path):
                return converter.convert(file_path)

        # If no converter found, provide helpful error message
        supported_formats = [f"{converter.__class__.__name__}" for converter in self.converters]
        raise ValueError(
            f"No converter found for file {file_path}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    def can_convert(self, file_path: str) -> bool:
        """
        Check if any converter can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if any converter can handle the file, False otherwise
        """
        for converter in self.converters:
            if converter.can_convert(file_path):
                return True
        return False

    def get_converter(self, file_path: str) -> BaseDataConverter:
        """
        Get the converter that can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            The first converter that can handle the file

        Raises:
            ValueError: If no converter can handle the file
        """
        for converter in self.converters:
            if converter.can_convert(file_path):
                return converter
        raise ValueError(f"No converter found for file {file_path}")

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported format names.

        Returns:
            List of converter class names that are currently registered
        """
        return [converter.__class__.__name__ for converter in self.converters]
