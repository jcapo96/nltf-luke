"""
Main analysis class that coordinates analysis across multiple datasets.

This class has been restructured to use specialized analysis classes,
making it more modular and maintainable.
"""

from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from .purity_analysis import PurityAnalysis
from .temperature_analysis import TemperatureAnalysis
from .h2o_analysis import H2OConcentrationAnalysis
from .liquid_level_analysis import LiquidLevelAnalysis


class Analysis:
    """
    Main analysis class that coordinates analysis across multiple datasets.

    This class has been restructured to use specialized analysis classes,
    making it more modular and maintainable.
    """

    def __init__(self, dataset_manager):
        """
        Initialize the main analysis class.

        Args:
            dataset_manager: DatasetManager instance containing the datasets to analyze
        """
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
        """
        Analyze purity across all datasets.

        Args:
            show: Whether to display the plot
            ax: Matplotlib axes to plot on
            fit_legend: Whether to show fit legend
            manual: Whether to use manual time selection
            integration_time_ini: Initial integration time in minutes
            integration_time_end: Final integration time in minutes
            offset_ini: Initial offset in minutes
            offset_end: Final offset in minutes

        Returns:
            Self for method chaining
        """
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
        """
        Analyze temperature across all datasets.

        Args:
            show: Whether to display the plot
            ax: Matplotlib axes to plot on
            manual: Whether to use manual time selection
            integration_time_ini: Initial integration time in minutes
            integration_time_end: Final integration time in minutes
            offset_ini: Initial offset in minutes
            offset_end: Final offset in minutes

        Returns:
            Self for method chaining
        """
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
        """
        Analyze H2O concentration across all datasets.

        Args:
            show: Whether to display the plot
            ax: Matplotlib axes to plot on
            manual: Whether to use manual time selection
            integration_time_ini: Initial integration time in minutes
            integration_time_end: Final integration time in minutes
            offset_ini: Initial offset in minutes
            offset_end: Final offset in minutes

        Returns:
            Self for method chaining
        """
        self.h2o_analysis.analyze(
            show=show, ax=ax, manual=manual,
            integration_time_ini=integration_time_ini,
            integration_time_end=integration_time_end,
            offset_ini=offset_ini, offset_end=offset_end
        )
        return self

    def level(self, show: bool = False, ax: Optional[plt.Axes] = None, manual: bool = False,
              fit_legend: bool = False) -> 'Analysis':
        """
        Analyze liquid level across all datasets.

        Args:
            show: Whether to display the plot
            ax: Matplotlib axes to plot on
            manual: Whether to use manual time selection
            fit_legend: Whether to show fit legend

        Returns:
            Self for method chaining
        """
        self.level_analysis.analyze(show=show, ax=ax, manual=manual, fit_legend=fit_legend)
        return self

    def get_analysis_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get results from all analysis types.

        Returns:
            Dictionary containing results from all analysis types
        """
        return {
            'purity': self.purity_analysis.get_results(),
            'temperature': self.temperature_analysis.get_results(),
            'h2o_concentration': self.h2o_analysis.get_results(),
            'liquid_level': self.level_analysis.get_results()
        }
