"""
iHistorian Converter for CSV files with semicolon-separated values.

This converter handles CSV files where all signals (liquid level, temperature,
purity, H2O concentration) are combined in a single file with sparse data
at different sampling rates.
"""

import pandas as pd
import os
from typing import Optional, Dict, Any
from datetime import datetime
import warnings

from .base_converter import BaseDataConverter
from core.standard_format import StandardDataFormat


class iHistorianConverter(BaseDataConverter):
    """
    Converter for iHistorian CSV data format.

    Expected structure:
    - Semicolon-separated CSV files
    - Header row with signal names
    - Combined signals in one file with sparse data
    - Files named: *_baseline.csv, *_liquid.csv, *_ullage.csv

    Signal column mapping:
    - category: timestamp column
    - PAB_S1.LT_13_AR_REAL.F_CV: liquid level
    - PAB_S1.TE_324_AR_REAL.F_CV: temperature
    - Luke.PRM_LIFETIME.F_CV: purity
    - PAB_S1.AE_611_AR_REAL.F_CV: H2O concentration
    """

    def __init__(self):
        self.name = "iHistorianConverter"
        self.supported_extensions = ['.csv']

        # Define the mapping between iHistorian column names and standard signal names
        self.signal_columns = {
            'PAB_S1.LT_13_AR_REAL.F_CV': 'liquid_level',
            'PAB_S1.TE_324_AR_REAL.F_CV': 'temperature',
            'Luke.PRM_LIFETIME.F_CV': 'purity',
            'PAB_S1.AE_611_AR_REAL.F_CV': 'h2o_concentration'
        }

    def can_convert(self, file_path: str) -> bool:
        """
        Check if this converter can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this converter can handle the file, False otherwise
        """
        if not os.path.exists(file_path):
            return False

        # Check file extension
        if not file_path.lower().endswith('.csv'):
            return False

        try:
            # Read first few lines to check format
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()

            # Check for semicolon delimiter
            if ';' not in first_line:
                return False

            # Check for expected column headers
            expected_headers = ['category', 'PAB_S1.LT_13_AR_REAL.F_CV']
            if not all(header in first_line for header in expected_headers):
                return False

            return True

        except Exception as e:
            warnings.warn(f"Error checking file format for {file_path}: {e}")
            return False

    def convert(self, file_path: str) -> Optional[StandardDataFormat]:
        """
        Convert iHistorian CSV file to StandardDataFormat.

        Args:
            file_path: Path to the CSV file

        Returns:
            StandardDataFormat object with converted data or None if conversion fails
        """
        try:
            if not self.can_convert(file_path):
                return None

            # Read CSV file with semicolon delimiter
            df = pd.read_csv(file_path, sep=';')

            if df.empty:
                warnings.warn(f"Empty file: {file_path}")
                return None

            # Extract dataset type from filename
            dataset_type = self.get_dataset_type(file_path)

            # Parse timestamp from category column
            if 'category' not in df.columns:
                warnings.warn(f"No 'category' column found in {file_path}")
                return None

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['category'], errors='coerce')

            # Remove rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])

            if df.empty:
                warnings.warn(f"No valid timestamps found in {file_path}")
                return None

            # Extract signals and create pandas Series
            liquid_level = self._extract_signal(df, 'PAB_S1.LT_13_AR_REAL.F_CV', 'liquid_level')
            temperature = self._extract_signal(df, 'PAB_S1.TE_324_AR_REAL.F_CV', 'temperature')
            purity = self._extract_signal(df, 'Luke.PRM_LIFETIME.F_CV', 'purity')
            h2o_concentration = self._extract_signal(df, 'PAB_S1.AE_611_AR_REAL.F_CV', 'h2o_concentration')

            # Use liquid level timestamp as primary reference (most reliable)
            primary_timestamp = liquid_level.index if liquid_level is not None and not liquid_level.empty else df['timestamp']

            # Create StandardDataFormat object
            return StandardDataFormat(
                timestamp=primary_timestamp,
                liquid_level=liquid_level,
                h2o_concentration=h2o_concentration,
                temperature=temperature,
                purity=purity,
                dataset_name=dataset_type,
                source_file=file_path
            )

        except Exception as e:
            warnings.warn(f"Failed to convert iHistorian file {file_path}: {e}")
            return None

    def _extract_signal(self, df: pd.DataFrame, column_name: str, signal_name: str) -> Optional[pd.Series]:
        """
        Extract a signal column and create a pandas Series with timestamp index.

        Args:
            df: DataFrame containing the data
            column_name: Name of the column to extract
            signal_name: Name for the resulting Series

        Returns:
            pandas Series with timestamp index or None if extraction fails
        """
        try:
            if column_name not in df.columns:
                warnings.warn(f"Column {column_name} not found in data")
                return None

            # Get non-empty values (filter out NaN and empty strings)
            mask = df[column_name].notna() & (df[column_name] != '') & (df[column_name] != ' ')

            if not mask.any():
                warnings.warn(f"No valid data found for {signal_name}")
                return None

            # Extract valid data with corresponding timestamps
            valid_data = df.loc[mask, [column_name, 'timestamp']]

            # Convert values to numeric, handling any conversion errors
            try:
                values = pd.to_numeric(valid_data[column_name], errors='coerce')
            except Exception:
                values = valid_data[column_name]

            # Remove any remaining NaN values after numeric conversion
            valid_mask = values.notna()
            if not valid_mask.any():
                warnings.warn(f"No valid numeric data found for {signal_name}")
                return None

            # Create Series with timestamp index
            signal_series = pd.Series(
                values[valid_mask].values,
                index=valid_data.loc[valid_mask, 'timestamp'],
                name=signal_name
            )

            return signal_series

        except Exception as e:
            warnings.warn(f"Error extracting signal {signal_name}: {e}")
            return None

    def get_dataset_type(self, file_path: str) -> str:
        """
        Extract dataset type from file path.

        Args:
            file_path: Path to the file

        Returns:
            String indicating dataset type: 'baseline', 'ullage', 'liquid', or 'unknown'
        """
        if not os.path.exists(file_path):
            return 'unknown'

        filename = os.path.basename(file_path).lower()

        if 'baseline' in filename:
            return 'baseline'
        elif 'ullage' in filename:
            return 'ullage'
        elif 'liquid' in filename:
            return 'liquid'
        else:
            return 'unknown'

    def get_metadata(self) -> Dict[str, Any]:
        """Get converter metadata."""
        return {
            'name': self.name,
            'description': 'Converter for iHistorian CSV files with semicolon-separated values',
            'supported_extensions': self.supported_extensions,
            'data_structure': {
                'format': 'Semicolon-separated CSV',
                'signals': 'Combined in single file with sparse data',
                'sampling': 'Different rates per signal'
            },
            'signal_mapping': self.signal_columns
        }
