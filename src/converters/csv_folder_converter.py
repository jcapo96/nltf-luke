"""
CSV Folder Converter for September2025 data format.

This converter handles data organized in subdirectories (baseline, ullage, liquid)
with separate CSV files for each signal type (ae=H2O, lt=level, prm=purity, te=temperature).
"""

import pandas as pd
import os
from typing import Optional, Dict, Any
from datetime import datetime
import warnings

from .base_converter import BaseDataConverter
from core.standard_format import StandardDataFormat


class CsvFolderConverter(BaseDataConverter):
    """
    Converter for CSV data organized in subdirectories.

    Expected structure:
    data_path/
    ├── baseline/
    │   ├── ae_*.csv (H2O concentration)
    │   ├── lt_*.csv (liquid level)
    │   ├── prm_*.csv (purity/lifetime)
    │   └── te_*.csv (temperature)
    ├── ullage/
    │   └── (same files as baseline)
    └── liquid/
        └── (same files as baseline)
    """

    def __init__(self):
        self.name = "CsvFolderConverter"
        self.supported_extensions = ['.csv']

    def can_convert(self, file_path: str) -> bool:
        """Check if this converter can handle the given file path."""
        if not os.path.exists(file_path):
            return False

        # Check if it's a directory with at least one of the expected subdirectory structure
        if os.path.isdir(file_path):
            # Check for case-insensitive directory names
            subdirs = [d.lower() for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]
            expected_dirs = ['baseline', 'ullage', 'liquid']
            # Return True if at least one expected directory is found
            return any(req_dir in subdirs for req_dir in expected_dirs)

        return False

    def convert(self, file_path: str) -> Optional[StandardDataFormat]:
        """
        Convert CSV folder data to StandardDataFormat.

        For CSV folder converter, this method loads the baseline dataset by default.
        The dataset manager will handle loading the other datasets separately.

        Args:
            file_path: Path to the data directory containing baseline/, ullage/, liquid/ subdirectories

        Returns:
            StandardDataFormat object for baseline data or None if conversion fails
        """
        try:
            if not self.can_convert(file_path):
                return None

            # Load baseline data by default
            baseline_data = self._load_dataset_data(os.path.join(file_path, 'baseline'))

            if not baseline_data:
                return None

            # Create StandardDataFormat for baseline data
            return self._create_standard_format(baseline_data, 'baseline', file_path)

        except Exception as e:
            warnings.warn(f"Failed to convert CSV folder data: {e}")
            return None

    def convert_dataset(self, file_path: str, dataset_type: str) -> Optional[StandardDataFormat]:
        """
        Convert a specific dataset from CSV folder data.

        Args:
            file_path: Path to the data directory containing baseline/, ullage/, liquid/ subdirectories
            dataset_type: Type of dataset to convert ('baseline', 'ullage', 'liquid')

        Returns:
            StandardDataFormat object or None if conversion fails
        """
        try:
            if not self.can_convert(file_path):
                return None

            # Find the actual directory name (case-insensitive)
            actual_dataset_dir = None
            for d in os.listdir(file_path):
                if os.path.isdir(os.path.join(file_path, d)) and d.lower() == dataset_type.lower():
                    actual_dataset_dir = d
                    break

            if not actual_dataset_dir:
                warnings.warn(f"Dataset directory '{dataset_type}' not found in '{file_path}'")
                return None

            # Load data from the specified subdirectory
            dataset_data = self._load_dataset_data(os.path.join(file_path, actual_dataset_dir))

            if not dataset_data:
                return None

            # Create StandardDataFormat for the dataset
            return self._create_standard_format(dataset_data, dataset_type, file_path)

        except Exception as e:
            warnings.warn(f"Failed to convert CSV folder dataset {dataset_type}: {e}")
            return None

    def _load_dataset_data(self, dataset_path: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load data from a single dataset directory (baseline, ullage, or liquid).

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Dictionary with signal data or None if loading fails
        """
        try:
            # Find CSV files in the directory
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

            if not csv_files:
                return None

            data = {}

            # Load each CSV file based on its prefix
            for csv_file in csv_files:
                file_path = os.path.join(dataset_path, csv_file)

                if csv_file.startswith('ae_'):
                    # H2O concentration data
                    df = self._load_csv_file(file_path)
                    if df is not None:
                        data['h2o_concentration'] = df

                elif csv_file.startswith('lt_'):
                    # Liquid level data
                    df = self._load_csv_file(file_path)
                    if df is not None:
                        data['liquid_level'] = df

                elif csv_file.startswith('prm_'):
                    # Purity/lifetime data
                    df = self._load_csv_file(file_path)
                    if df is not None:
                        data['purity'] = df

                elif csv_file.startswith('te_'):
                    # Temperature data
                    df = self._load_csv_file(file_path)
                    if df is not None:
                        data['temperature'] = df

            return data if data else None

        except Exception as e:
            warnings.warn(f"Failed to load dataset from {dataset_path}: {e}")
            return None

    def _load_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load and process a single CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Processed DataFrame or None if loading fails
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            if df.empty:
                return None

            # Check for required columns
            required_cols = ['TimeStamp', 'Value']
            if not all(col in df.columns for col in required_cols):
                return None

            # Parse timestamps
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

            # Filter for good data quality if available
            if 'DataQuality' in df.columns:
                df = df[df['DataQuality'] == 'Good']

            # Sort by timestamp
            df = df.sort_values('TimeStamp').reset_index(drop=True)

            # Create a clean DataFrame with just timestamp and value
            result_df = pd.DataFrame({
                'timestamp': df['TimeStamp'],
                'value': df['Value']
            })

            return result_df

        except Exception as e:
            warnings.warn(f"Failed to load CSV file {file_path}: {e}")
            return None

    def _create_standard_format(self, dataset_data: Dict[str, pd.DataFrame], dataset_type: str, file_path: str) -> StandardDataFormat:
        """
        Create a StandardDataFormat object from dataset data.

        Args:
            dataset_data: Dictionary containing signal data
            dataset_type: Type of dataset ('baseline', 'ullage', 'liquid')
            file_path: Original file path for metadata

        Returns:
            StandardDataFormat object
        """
        # Extract the primary timestamp from liquid level data (most reliable)
        if 'liquid_level' in dataset_data and not dataset_data['liquid_level'].empty:
            primary_timestamp = dataset_data['liquid_level']['timestamp']
        elif 'h2o_concentration' in dataset_data and not dataset_data['h2o_concentration'].empty:
            primary_timestamp = dataset_data['h2o_concentration']['timestamp']
        elif 'temperature' in dataset_data and not dataset_data['temperature'].empty:
            primary_timestamp = dataset_data['temperature']['timestamp']
        else:
            # Fallback to any available timestamp
            for signal_data in dataset_data.values():
                if not signal_data.empty and 'timestamp' in signal_data.columns:
                    primary_timestamp = signal_data['timestamp']
                    break
            else:
                raise ValueError("No valid timestamp data found")

        # Extract signal data and create Series with timestamp index
        liquid_level = None
        if 'liquid_level' in dataset_data and not dataset_data['liquid_level'].empty:
            liquid_level_df = dataset_data['liquid_level']
            liquid_level = pd.Series(
                liquid_level_df['value'].values,
                index=pd.to_datetime(liquid_level_df['timestamp']),
                name='liquid_level'
            )

        h2o_concentration = None
        if 'h2o_concentration' in dataset_data and not dataset_data['h2o_concentration'].empty:
            h2o_df = dataset_data['h2o_concentration']
            h2o_concentration = pd.Series(
                h2o_df['value'].values,
                index=pd.to_datetime(h2o_df['timestamp']),
                name='h2o_concentration'
            )

        temperature = None
        if 'temperature' in dataset_data and not dataset_data['temperature'].empty:
            temp_df = dataset_data['temperature']
            temperature = pd.Series(
                temp_df['value'].values,
                index=pd.to_datetime(temp_df['timestamp']),
                name='temperature'
            )

        purity = None
        if 'purity' in dataset_data and not dataset_data['purity'].empty:
            purity_df = dataset_data['purity']
            purity = pd.Series(
                purity_df['value'].values,
                index=pd.to_datetime(purity_df['timestamp']),
                name='purity'
            )

        return StandardDataFormat(
            timestamp=primary_timestamp,
            liquid_level=liquid_level,
            h2o_concentration=h2o_concentration,
            temperature=temperature,
            purity=purity,
            dataset_name=dataset_type,
            source_file=file_path
        )

    def get_dataset_type(self, file_path: str) -> str:
        """
        Extract dataset type from file path.

        Args:
            file_path: Path to the file or directory

        Returns:
            String indicating dataset type: 'baseline', 'ullage', 'liquid', or 'unknown'
        """
        if not os.path.exists(file_path):
            return 'unknown'

        # For directory paths, check the parent directory name
        if os.path.isdir(file_path):
            dir_name = os.path.basename(file_path).lower()
            if dir_name in ['baseline', 'ullage', 'liquid']:
                return dir_name
            return 'unknown'

        # For file paths, check the parent directory
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        if parent_dir in ['baseline', 'ullage', 'liquid']:
            return parent_dir

        return 'unknown'

    def get_metadata(self) -> Dict[str, Any]:
        """Get converter metadata."""
        return {
            'name': self.name,
            'description': 'Converter for CSV data organized in subdirectories (baseline, ullage, liquid)',
            'supported_extensions': self.supported_extensions,
            'data_structure': {
                'baseline/': 'Baseline (no sample) data',
                'ullage/': 'Ullage (gas phase) data',
                'liquid/': 'Liquid (submerged) data'
            },
            'signal_files': {
                'ae_*.csv': 'H2O concentration data',
                'lt_*.csv': 'Liquid level data',
                'prm_*.csv': 'Purity/lifetime data',
                'te_*.csv': 'Temperature data'
            }
        }
