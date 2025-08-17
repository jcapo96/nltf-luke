"""
Dataset class for handling data using the new data format abstraction layer.

This class now works with StandardDataFormat instead of hardcoded column names,
making it format-independent and more flexible.
"""

from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from core.standard_format import StandardDataFormat
from processors.liquid_level_processor import LiquidLevelProcessor
from processors.h2o_processor import H2OConcentrationProcessor
from processors.temperature_processor import TemperatureProcessor
from processors.purity_processor import PurityProcessor


class Dataset:
    """
    A class to handle datasets using the new data format abstraction layer.

    This class now works with StandardDataFormat instead of hardcoded column names,
    making it format-independent and more flexible.
    """

    def __init__(self, path: str, name: Optional[str] = None):
        """
        Initialize the dataset.

        Args:
            path: Path to the data file
            name: Optional name for the dataset
        """
        self.path = path
        self.name = name

        # Data storage using standard format
        self.standard_data: Optional[StandardDataFormat] = None

        # Processors
        self.liquid_level: Optional[LiquidLevelProcessor] = None
        self.h2o_concentration: Optional[H2OConcentrationProcessor] = None
        self.temperature: Optional[TemperatureProcessor] = None
        self.purity: Optional[PurityProcessor] = None

        # Analysis results
        self._analysis_results: Dict[str, Any] = {}

    def load(self, format_manager=None) -> 'Dataset':
        """
        Load the dataset from the specified path using the format manager.

        Args:
            format_manager: Optional format manager instance

        Returns:
            Self for method chaining
        """
        try:
            if format_manager is None:
                from converters.data_format_manager import DataFormatManager
                format_manager = DataFormatManager()

            # Convert to standard format
            self.standard_data = format_manager.convert_file(self.path)
            self.name = self.standard_data.dataset_name

            # Initialize processors
            self._initialize_processors()

        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.standard_data = None

        return self

    def _initialize_processors(self):
        """Initialize data processors after data is loaded."""
        if self.standard_data is not None:
            try:
                self.liquid_level = LiquidLevelProcessor(self.standard_data)
                self.h2o_concentration = H2OConcentrationProcessor(self.standard_data)
                self.temperature = TemperatureProcessor(self.standard_data)
                self.purity = PurityProcessor(self.standard_data)
            except Exception as e:
                print(f"Warning: Error initializing processors: {e}")
                self.liquid_level = None
                self.h2o_concentration = None
                self.temperature = None
                self.purity = None
        else:
            # Set processors to None if no data
            self.liquid_level = None
            self.h2o_concentration = None
            self.temperature = None
            self.purity = None

    def describe(self) -> None:
        """Print a description of the dataset."""
        if self.standard_data is not None:
            print(f"Dataset: {self.name}")
            print(f"Source: {self.standard_data.source_file}")
            print(f"Data points: {len(self.standard_data.timestamp)}")
            print(f"Time range: {self.standard_data.timestamp.min()} to {self.standard_data.timestamp.max()}")

            # Show available data types
            available_data = []
            if self.standard_data.liquid_level is not None:
                available_data.append("Liquid Level")
            if self.standard_data.h2o_concentration is not None:
                available_data.append("H2O Concentration")
            if self.standard_data.temperature is not None:
                available_data.append("Temperature")
            if self.standard_data.purity is not None:
                available_data.append("Purity")

            print(f"Available data types: {', '.join(available_data)}")
        else:
            print("No data available. Please load the dataset first.")

    def show(self) -> None:
        """Plot the data in the dataset."""
        if self.standard_data is None:
            print("No data to show. Please load the dataset first.")
            return

        # Create subplots for available data
        available_plots = []
        if self.standard_data.liquid_level is not None:
            available_plots.append("Liquid Level")
        if self.standard_data.h2o_concentration is not None:
            available_plots.append("H2O Concentration")
        if self.standard_data.temperature is not None:
            available_plots.append("Temperature")
        if self.standard_data.purity is not None:
            available_plots.append("Purity")

        if not available_plots:
            print("No data available to plot.")
            return

        n_plots = len(available_plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        fig.suptitle(f"{self.name} Dataset", fontsize=16, fontweight='bold')

        start_time = self.standard_data.timestamp.min()
        end_time = self.standard_data.timestamp.max()

        for i, plot_type in enumerate(available_plots):
            ax = axes[i]

            if plot_type == "Liquid Level" and self.liquid_level:
                self.liquid_level.plot_level(start_time, end_time, ax=ax, dataset_name=self.name)
            elif plot_type == "H2O Concentration" and self.h2o_concentration:
                h2o_data = self.h2o_concentration.calculate_h2o_concentration(start_time, end_time)
                self.h2o_concentration.plot_h2o_concentration(start_time, end_time, h2o_data, ax=ax, dataset_name=self.name)
            elif plot_type == "Temperature" and self.temperature:
                temp_data = self.temperature.calculate_temperature(start_time, end_time)
                self.temperature.plot_temperature(start_time, end_time, temp_data, ax=ax, dataset_name=self.name)
            elif plot_type == "Purity" and self.purity:
                purity_data = self.purity.calculate_purity(start_time, end_time)
                self.purity.plot_purity(start_time, end_time, purity_data, ax=ax, dataset_name=self.name)

        plt.tight_layout()
        plt.show()

    def get_analysis_results(self) -> Dict[str, Any]:
        """
        Get analysis results for this dataset.

        Returns:
            Copy of the analysis results dictionary
        """
        return self._analysis_results.copy()

    def set_analysis_results(self, analysis_type: str, results: Dict[str, Any]):
        """
        Set analysis results for this dataset.

        Args:
            analysis_type: Type of analysis performed
            results: Results dictionary
        """
        self._analysis_results[analysis_type] = results
