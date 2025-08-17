"""
Converts Excel files from Seeq new format to StandardDataFormat.
"""

import os
import pandas as pd
from .base_converter import BaseDataConverter
from core.standard_format import StandardDataFormat


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
                        # Signal found successfully
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
