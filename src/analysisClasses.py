from dataClasses import Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
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
    """Manages multiple datasets for analysis."""

    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        self.datasets: Dict[str, Dataset] = {}
        self._load_datasets()

    def _load_datasets(self):
        """Load all required datasets."""
        dataset_types = ['baseline', 'ullage', 'liquid']

        for dataset_type in dataset_types:
            try:
                file_path = f"{self.path}/{self.name}_{dataset_type}.xlsx"
                self.datasets[dataset_type] = Dataset(path=file_path, name=dataset_type.capitalize())
            except Exception as e:
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
                dataset = self.datasets[dataset_type]
                dataset.load()

                if dataset.liquid_level is None:
                    # Dataset missing liquid level data - skip
                    continue

                start_time, end_time = dataset.liquid_level.find_times(manual=manual)[4:6]

                prepared_data[dataset_type] = {
                    'dataset': dataset,
                    'times': dataset.liquid_level.find_times(manual=manual)
                }

            except Exception as e:
                # Error preparing dataset - continue with others
                continue

        return prepared_data


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
                fit_legend: bool = False, manual: bool = False) -> 'PurityAnalysis':
        """Analyze purity across all datasets."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        prepared_data = self.dataset_manager.prepare_datasets(manual=manual)

        for dataset_type, data in prepared_data.items():
            try:
                dataset = data['dataset']
                start_time, end_time = data['times'][4], data['times'][5]  # start_time and end_time

                if dataset.purity is not None:
                    purity_data = dataset.purity.calculate_purity(start_time, end_time)
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
                manual: bool = False) -> 'TemperatureAnalysis':
        """Analyze temperature across all datasets."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        prepared_data = self.dataset_manager.prepare_datasets(manual=manual)
        plotted_datasets = []

        for dataset_type, data in prepared_data.items():
            try:
                dataset = data['dataset']
                start_time, end_time = data['times'][4], data['times'][5]

                if dataset.temperature is not None:
                    temp_data = dataset.temperature.calculate_temperature(start_time, end_time)
                    dataset.temperature.plot_temperature(
                        start_time, end_time, temp_data,
                        ax=ax, color=self.colors[dataset_type], dataset_name=dataset_type.capitalize()
                    )
                    plotted_datasets.append(dataset_type)

                    # Store results
                    self._results[dataset_type] = {
                        'temperature_data': temp_data,
                        'start_time': start_time,
                        'end_time': end_time
                    }

            except Exception as e:
                # Error analyzing dataset - continue with others
                pass

        if plotted_datasets:
            ax.legend(ncol=len(plotted_datasets))

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
                start_time, end_time = data['times'][4], data['times'][5]

                if dataset.h2o_concentration is not None:
                    h2o_data = dataset.h2o_concentration.calculate_concentration(
                        start_time, end_time,
                        integration_time_ini, integration_time_end,
                        offset_ini, offset_end
                    )
                    dataset.h2o_concentration.plot_concentration(
                        start_time, end_time, h2o_data,
                        ax=ax, color=self.colors[dataset_type], dataset_name=dataset_type.capitalize()
                    )

                    # Store results
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

    def analyze(self, ax: Optional[plt.Axes] = None, manual: bool = False,
                fit_legend: bool = False) -> 'LiquidLevelAnalysis':
        """Analyze liquid level across all datasets."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        prepared_data = self.dataset_manager.prepare_datasets(manual=manual)

        for dataset_type, data in prepared_data.items():
            try:
                dataset = data['dataset']
                start_time, end_time = data['times'][4], data['times'][5]

                if dataset.liquid_level is not None:
                    dataset.liquid_level.plot_level(
                        start_time, end_time,
                        ax=ax, color=self.colors[dataset_type],
                        fit_legend=fit_legend, dataset_name=dataset_type.capitalize()
                    )

                    # Store results
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

    def __init__(self, path: str, name: Optional[str] = None):
        self.dataset_manager = DatasetManager(path, name)
        self.name = name

        # Initialize specialized analysis classes
        self.purity_analysis = PurityAnalysis(self.dataset_manager)
        self.temperature_analysis = TemperatureAnalysis(self.dataset_manager)
        self.h2o_analysis = H2OConcentrationAnalysis(self.dataset_manager)
        self.level_analysis = LiquidLevelAnalysis(self.dataset_manager)

    def purity(self, show: bool = False, ax: Optional[plt.Axes] = None,
               fit_legend: bool = False, manual: bool = False) -> 'Analysis':
        """Analyze purity across all datasets."""
        self.purity_analysis.analyze(show=show, ax=ax, fit_legend=fit_legend, manual=manual)
        return self

    def temperature(self, show: bool = False, ax: Optional[plt.Axes] = None,
                   manual: bool = False) -> 'Analysis':
        """Analyze temperature across all datasets."""
        self.temperature_analysis.analyze(show=show, ax=ax, manual=manual)
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

    def level(self, ax: Optional[plt.Axes] = None, manual: bool = False,
              fit_legend: bool = False) -> 'Analysis':
        """Analyze liquid level across all datasets."""
        self.level_analysis.analyze(ax=ax, manual=manual, fit_legend=fit_legend)
        return self

    def get_analysis_results(self) -> Dict[str, Dict[str, Any]]:
        """Get results from all analysis types."""
        return {
            'purity': self.purity_analysis.get_results(),
            'temperature': self.temperature_analysis.get_results(),
            'h2o_concentration': self.h2o_analysis.get_results(),
            'liquid_level': self.level_analysis.get_results()
        }