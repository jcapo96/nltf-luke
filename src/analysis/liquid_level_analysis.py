"""
Analyzes liquid level data across multiple datasets.
"""

from typing import Optional
import matplotlib.pyplot as plt
from .base_analysis import BaseAnalysis


class LiquidLevelAnalysis(BaseAnalysis):
    """
    Analyzes liquid level data across multiple datasets.

    This analysis class coordinates liquid level analysis across baseline, ullage, and liquid datasets.
    It uses the liquid level processor to determine run start/end times and then plots
    liquid level data within those time windows.
    """

    def __init__(self, dataset_manager):
        """
        Initialize the liquid level analysis.

        Args:
            dataset_manager: DatasetManager instance containing the datasets to analyze
        """
        super().__init__()
        self.dataset_manager = dataset_manager
        self.colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }

    def analyze(self, show: bool = False, ax: Optional[plt.Axes] = None, manual: bool = False,
                fit_legend: bool = False) -> 'LiquidLevelAnalysis':
        """
        Analyze liquid level across all datasets.

        Args:
            show: Whether to display the plot (not used in current implementation)
            ax: Matplotlib axes to plot on (creates new one if None)
            manual: Whether to use manual time selection
            fit_legend: Whether to show fit legend (not used in current implementation)

        Returns:
            Self for method chaining
        """
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

        # Only add legend if there are labeled artists
        if ax.get_legend_handles_labels()[0]:
            ax.legend(ncol=1)
        return self
