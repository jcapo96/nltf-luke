"""
DatasetManager class for managing multiple datasets and providing a unified interface for analysis.

This class handles loading, validation, and preparation of multiple datasets
from different sources and formats.
"""

import os
import json
from typing import Dict, Optional, Any, List
from .dataset import Dataset
from converters.data_format_manager import DataFormatManager


class DatasetManager:
    """
    Manages multiple datasets and provides a unified interface for analysis.

    This class handles loading, validation, and preparation of multiple datasets
    from different sources and formats.
    """

    def __init__(self, dataset_paths: Dict[str, str], name: str, config_file: str = None):
        """
        Initialize the dataset manager.

        Args:
            dataset_paths: Dictionary mapping dataset types to file paths
            name: Name for this dataset collection
            config_file: Path to JSON configuration file (optional)
        """
        self.dataset_paths = dataset_paths
        self.name = name
        self.datasets: Dict[str, Dataset] = {}
        self.format_manager = DataFormatManager()
        self.config_file = config_file
        self.preferred_converter = self._get_preferred_converter()
        self._load_datasets()

    def _load_datasets(self):
        """Load all required datasets using the format manager and preferred converter."""
        # Check if preferred converter is available
        if self.preferred_converter:
            # Verify the preferred converter exists
            converter_exists = any(
                converter.__class__.__name__ == self.preferred_converter
                for converter in self.format_manager.converters
            )

            if not converter_exists:
                print(f"Warning: Preferred converter '{self.preferred_converter}' not found, falling back to SeeqNewConverter...")
                self.preferred_converter = "SeeqNewConverter"
            # Converter preference set successfully
        else:
            print("Using default converter: SeeqNewConverter")
            self.preferred_converter = "SeeqNewConverter"

        for dataset_type, file_path in self.dataset_paths.items():
            try:
                # Check if the format manager can handle this file
                if self.format_manager.can_convert(file_path):
                    # Create and load the dataset
                    dataset = Dataset(path=file_path, name=dataset_type.capitalize())
                    dataset.load(self.format_manager)
                    self.datasets[dataset_type] = dataset
                else:
                    print(f"Warning: No converter found for {file_path}")
            except Exception as e:
                print(f"Warning: Could not load dataset {dataset_type}: {e}")
                # Could not load dataset - continue with others
                pass

    def get_dataset(self, dataset_type: str) -> Optional[Dataset]:
        """
        Get a specific dataset by type.

        Args:
            dataset_type: Type of dataset to retrieve

        Returns:
            Dataset instance if found, None otherwise
        """
        return self.datasets.get(dataset_type)

    def get_all_datasets(self) -> Dict[str, Dataset]:
        """
        Get all loaded datasets.

        Returns:
            Copy of the datasets dictionary
        """
        return self.datasets.copy()

    def prepare_datasets(self, manual: bool = False) -> Dict[str, Any]:
        """
        Prepare all datasets for analysis by loading data and finding times.

        Args:
            manual: Whether to use manual time selection

        Returns:
            Dictionary containing prepared dataset data
        """
        prepared_data = {}

        for dataset_type, dataset in self.datasets.items():
            try:
                # Load the dataset using the format manager
                dataset.load(self.format_manager)

                if dataset.liquid_level is None:
                    # Dataset missing liquid level data - skip
                    continue

                # Get the times from find_times() method
                times = dataset.liquid_level.find_times(manual=manual)
                start_time = times['start_time']
                end_time = times['end_time']

                prepared_data[dataset_type] = {
                    'dataset': dataset,
                    'times': times
                }

            except Exception as e:
                # Error preparing dataset - continue with others
                continue

        return prepared_data

    def register_converter(self, converter):
        """
        Register a new data converter with the format manager.

        Args:
            converter: Converter instance to register
        """
        self.format_manager.register_converter(converter)

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported format names.

        Returns:
            List of supported format names
        """
        return self.format_manager.get_supported_formats()

    def _get_preferred_converter(self) -> Optional[str]:
        """
        Loads converter preferences from a JSON configuration file.
        Reads the 'Converter' key from the 'Data' section.
        If no config file is provided, it returns None.

        Returns:
            Preferred converter name if specified, None otherwise
        """
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Look for Converter in the Data section
                    if 'Data' in config and 'Converter' in config['Data']:
                        return config['Data']['Converter']
            except Exception as e:
                print(f"Warning: Could not load converter preference from {self.config_file}: {e}")
        return None
