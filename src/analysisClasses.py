import os
import json
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
try:
    from .dataFormats import DataFormatManager, StandardDataFormat
    from .dataClasses import Dataset
except ImportError:
    # Fallback for when running as script
    from dataFormats import DataFormatManager, StandardDataFormat
    from dataClasses import Dataset
from abc import ABC, abstractmethod


class BaseAnalysis(ABC):
    """Abstract base class for analysis operations."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._results: Dict[str, Any] = {}

    @abstractmethod
    def analyze(self, **kwargs) -> Any:
        """Perform the analysis operation."""
        pass

    def get_results(self) -> Dict[str, Any]:
        """Get the analysis results."""
        return self._results.copy()


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
                print(f"Warning: Preferred converter '{self.preferred_converter}' not found.")
                print(f"Available converters: {[c.__class__.__name__ for c in self.format_manager.converters]}")
                print("Falling back to SeeqNewConverter...")
                self.preferred_converter = "SeeqNewConverter"
            else:
                print(f"Using preferred converter: {self.preferred_converter}")
        else:
            print("No converter preference specified in config file.")
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
        """Get a specific dataset by type."""
        return self.datasets.get(dataset_type)

    def get_all_datasets(self) -> Dict[str, Dataset]:
        """Get all loaded datasets."""
        return self.datasets.copy()

    def prepare_datasets(self, manual: bool = False) -> Dict[str, Any]:
        """Prepare all datasets for analysis by loading data and finding times."""
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
        """Register a new data converter with the format manager."""
        self.format_manager.register_converter(converter)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported format names."""
        return self.format_manager.get_supported_formats()

    def _get_preferred_converter(self) -> Optional[str]:
        """
        Loads converter preferences from a JSON configuration file.
        Reads the 'Converter' key from the 'Data' section.
        If no config file is provided, it returns None.
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


class PurityAnalysis(BaseAnalysis):
    """Analyzes electron lifetime (purity) data across multiple datasets."""

    def __init__(self, dataset_manager: DatasetManager):
        super().__init__()
        self.dataset_manager = dataset_manager
        self.colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }

    def analyze(self, show: bool = False, ax: Optional[plt.Axes] = None,
                manual: bool = False, integration_time_ini: int = 60,
                integration_time_end: int = 480, offset_ini: int = 60,
                offset_end: int = 0, fit_legend: bool = False) -> 'PurityAnalysis':
        """Analyze purity across all datasets."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        prepared_data = self.dataset_manager.prepare_datasets(manual=manual)

        for dataset_type, data in prepared_data.items():
            try:
                dataset = data['dataset']

                # For purity analysis, use the find_times() method from liquid level processor
                if dataset.purity is not None and dataset.liquid_level is not None:
                    # Get the actual run start/end times from liquid level analysis
                    level_times = dataset.liquid_level.find_times()
                    start_time = level_times['start_time']
                    end_time = level_times['end_time']

                    if start_time is None or end_time is None:
                        # Fallback to dataset's own time range
                        purity_timestamp = dataset.standard_data.purity.index
                        start_time = purity_timestamp.min()
                        end_time = purity_timestamp.max()

                    purity_data = dataset.purity.calculate_purity(
                        start_time, end_time,
                        integration_time_ini=integration_time_ini,
                        integration_time_end=integration_time_end,
                        offset_ini=offset_ini,
                        offset_end=offset_end
                    )

                    dataset.purity.plot_purity(
                        start_time, end_time, purity_data,
                        ax=ax, color=self.colors[dataset_type],
                        fit_legend=fit_legend, dataset_name=dataset_type.capitalize()
                    )

                    # Store results for reporting
                    self._results[dataset_type] = {
                        'purity_data': purity_data,
                        'start_time': start_time,
                        'end_time': end_time
                    }

            except Exception as e:
                # Error analyzing dataset - continue with others
                pass

        return self


class TemperatureAnalysis(BaseAnalysis):
    """Analyzes temperature data across multiple datasets."""

    def __init__(self, dataset_manager: DatasetManager):
        super().__init__()
        self.dataset_manager = dataset_manager
        self.colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }

    def analyze(self, show: bool = False, ax: Optional[plt.Axes] = None,
                manual: bool = False, integration_time_ini: int = 60,
                integration_time_end: int = 480, offset_ini: int = 60,
                offset_end: int = 0) -> 'TemperatureAnalysis':
        """Analyze temperature across all datasets."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        prepared_data = self.dataset_manager.prepare_datasets(manual=manual)
        plotted_datasets = []

        for dataset_type, data in prepared_data.items():
            try:
                dataset = data['dataset']

                # For temperature analysis, use the find_times() method from liquid level processor
                if dataset.temperature is not None and dataset.liquid_level is not None:
                    # Get the actual run start/end times from liquid level analysis
                    level_times = dataset.liquid_level.find_times()
                    start_time = level_times['start_time']
                    end_time = level_times['end_time']

                    if start_time is None or end_time is None:
                        # Fallback to dataset's own time range
                        temp_timestamp = dataset.standard_data.temperature.index
                        start_time = temp_timestamp.min()
                        end_time = temp_timestamp.max()

                    temp_data = dataset.temperature.calculate_temperature(
                        start_time, end_time,
                        integration_time_ini=integration_time_ini,
                        integration_time_end=integration_time_end,
                        offset_ini=offset_ini,
                        offset_end=offset_end
                    )

                    dataset.temperature.plot_temperature(
                        start_time, end_time, temp_data,
                        ax=ax, color=self.colors[dataset_type],
                        dataset_name=dataset_type.capitalize()
                    )

                    # Track which datasets were plotted
                    plotted_datasets.append(dataset_type)

                    # Store results for reporting
                    self._results[dataset_type] = {
                        'temp_data': temp_data,
                        'start_time': start_time,
                        'end_time': end_time
                    }

            except Exception as e:
                # Error analyzing dataset - continue with others
                pass

        if plotted_datasets:
            ax.legend(ncol=len(plotted_datasets))
        else:
            ax.legend(ncol=1)

        return self


class H2OConcentrationAnalysis(BaseAnalysis):
    """Analyzes H2O concentration data across multiple datasets."""

    def __init__(self, dataset_manager: DatasetManager):
        super().__init__()
        self.dataset_manager = dataset_manager
        self.colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }

    def analyze(self, show: bool = False, ax: Optional[plt.Axes] = None,
                manual: bool = False, integration_time_ini: int = 60,
                integration_time_end: int = 60, offset_ini: int = 1,
                offset_end: int = 8) -> 'H2OConcentrationAnalysis':
        """Analyze H2O concentration across all datasets."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        prepared_data = self.dataset_manager.prepare_datasets(manual=manual)

        for dataset_type, data in prepared_data.items():
            try:
                dataset = data['dataset']

                # For H2O analysis, use the find_times() method from liquid level processor
                if dataset.h2o_concentration is not None and dataset.liquid_level is not None:
                    # Get the actual run start/end times from liquid level analysis
                    level_times = dataset.liquid_level.find_times()
                    start_time = level_times['start_time']
                    end_time = level_times['end_time']

                    if start_time is None or end_time is None:
                        # Fallback to dataset's own time range
                        h2o_timestamp = dataset.standard_data.h2o_concentration.index
                        start_time = h2o_timestamp.min()
                        end_time = h2o_timestamp.max()

                    h2o_data = dataset.h2o_concentration.calculate_h2o_concentration(
                        start_time, end_time,
                        integration_time_ini=integration_time_ini,
                        integration_time_end=integration_time_end,
                        offset_ini=offset_ini,
                        offset_end=offset_end
                    )

                    dataset.h2o_concentration.plot_concentration(
                        start_time, end_time, h2o_data,
                        ax=ax, color=self.colors[dataset_type],
                        dataset_name=dataset_type.capitalize()
                    )

                    # Store results for reporting
                    self._results[dataset_type] = {
                        'h2o_data': h2o_data,
                        'start_time': start_time,
                        'end_time': end_time
                    }

            except Exception as e:
                # Error analyzing dataset - continue with others
                pass

        ax.set_ylim(0, None)
        ax.legend(ncol=3)
        # self._print_h2o_summary() # Removed as per edit hint

        return self

    # def _print_h2o_summary(self): # Removed as per edit hint
    #     """Print a summary of H2O concentration analysis results.""" # Removed as per edit hint
    #     print("H2O Concentration Analysis Summary:") # Removed as per edit hint
    #     print("-" * 50) # Removed as per edit hint
    #     for dataset_type, results in self._results.items(): # Removed as per edit hint
    #         if 'h2o_data' in results: # Removed as per edit hint
    #             h2o_data = results['h2o_data'] # Removed as per edit hint
    #             initial_h2o = h2o_data['initial'] # Removed as per edit hint
    #             initial_h2o_err = h2o_data['initial_error'] # Removed as per edit hint
    #             final_h2o = h2o_data['final'] # Removed as per edit hint
    #             final_h2o_err = h2o_data['final_error'] # Removed as per edit hint
    #             delta_h2o = final_h2o - initial_h2o # Removed as per edit hint
    #             delta_h2o_err = np.sqrt(final_h2o_err**2 + initial_h2o_err**2) # Removed as per edit hint
    #             print(f"{dataset_type.capitalize()}:") # Removed as per edit hint
    #             print(f"  Initial H2O: {initial_h2o:.2f} ± {initial_h2o_err:.2f} ppb") # Removed as per edit hint
    #             print(f"  Final H2O: {final_h2o:.2f} ± {final_h2o_err:.2f} ppb") # Removed as per edit hint
    #             print(f"  ΔH2O: {delta_h2o:.2f} ± {delta_h2o_err:.2f} ppb") # Removed as per edit hint
    #             print() # Removed as per edit hint


class LiquidLevelAnalysis(BaseAnalysis):
    """Analyzes liquid level data across multiple datasets."""

    def __init__(self, dataset_manager: DatasetManager):
        super().__init__()
        self.dataset_manager = dataset_manager
        self.colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }

    def analyze(self, show: bool = False, ax: Optional[plt.Axes] = None, manual: bool = False,
                fit_legend: bool = False) -> 'LiquidLevelAnalysis':
        """Analyze liquid level across all datasets."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        prepared_data = self.dataset_manager.prepare_datasets(manual=manual)

        for dataset_type, data in prepared_data.items():
            try:
                dataset = data['dataset']

                # For liquid level analysis, use the find_times() method from liquid level processor
                if dataset.liquid_level is not None and dataset.liquid_level is not None:
                    # Get the actual run start/end times from liquid level analysis
                    level_times = dataset.liquid_level.find_times()
                    start_time = level_times['start_time']
                    end_time = level_times['end_time']

                    if start_time is None or end_time is None:
                        # Fallback to dataset's own time range
                        level_timestamp = dataset.standard_data.liquid_level.index
                        start_time = level_timestamp.min()
                        end_time = level_timestamp.max()

                    dataset.liquid_level.plot_level(
                        start_time, end_time,
                        ax=ax, color=self.colors[dataset_type],
                        dataset_name=dataset_type.capitalize()
                    )

                    # Store results for reporting
                    self._results[dataset_type] = {
                        'start_time': start_time,
                        'end_time': end_time
                    }

            except Exception as e:
                # Error analyzing dataset - continue with others
                pass

        ax.legend(ncol=1)
        return self


class Analysis:
    """
    Main analysis class that coordinates analysis across multiple datasets.

    This class has been restructured to use specialized analysis classes,
    making it more modular and maintainable.
    """

    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.name = dataset_manager.name

        # Initialize specialized analysis classes
        self.purity_analysis = PurityAnalysis(self.dataset_manager)
        self.temperature_analysis = TemperatureAnalysis(self.dataset_manager)
        self.h2o_analysis = H2OConcentrationAnalysis(self.dataset_manager)
        self.level_analysis = LiquidLevelAnalysis(self.dataset_manager)

    def purity(self, show: bool = False, ax: Optional[plt.Axes] = None,
               fit_legend: bool = False, manual: bool = False,
               integration_time_ini: int = 60, integration_time_end: int = 480,
               offset_ini: int = 60, offset_end: int = 0) -> 'Analysis':
        """Analyze purity across all datasets."""
        self.purity_analysis.analyze(
            show=show, ax=ax, fit_legend=fit_legend, manual=manual,
            integration_time_ini=integration_time_ini,
            integration_time_end=integration_time_end,
            offset_ini=offset_ini, offset_end=offset_end
        )
        return self

    def temperature(self, show: bool = False, ax: Optional[plt.Axes] = None,
                   manual: bool = False, integration_time_ini: int = 60,
                   integration_time_end: int = 480, offset_ini: int = 60,
                   offset_end: int = 0) -> 'Analysis':
        """Analyze temperature across all datasets."""
        self.temperature_analysis.analyze(
            show=show, ax=ax, manual=manual,
            integration_time_ini=integration_time_ini,
            integration_time_end=integration_time_end,
            offset_ini=offset_ini, offset_end=offset_end
        )
        return self

    def h2oConcentration(self, show: bool = False, ax: Optional[plt.Axes] = None,
                         manual: bool = False, integration_time_ini: int = 60,
                         integration_time_end: int = 60, offset_ini: int = 1,
                         offset_end: int = 8) -> 'Analysis':
        """Analyze H2O concentration across all datasets."""
        self.h2o_analysis.analyze(
            show=show, ax=ax, manual=manual,
            integration_time_ini=integration_time_ini,
            integration_time_end=integration_time_end,
            offset_ini=offset_ini, offset_end=offset_end
        )
        return self

    def level(self, show: bool = False, ax: Optional[plt.Axes] = None, manual: bool = False,
              fit_legend: bool = False) -> 'Analysis':
        """Analyze liquid level across all datasets."""
        self.level_analysis.analyze(show=show, ax=ax, manual=manual, fit_legend=fit_legend)
        return self

    def get_analysis_results(self) -> Dict[str, Dict[str, Any]]:
        """Get results from all analysis types."""
        return {
            'purity': self.purity_analysis.get_results(),
            'temperature': self.temperature_analysis.get_results(),
            'h2o_concentration': self.h2o_analysis.get_results(),
            'liquid_level': self.level_analysis.get_results()
        }