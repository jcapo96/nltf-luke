"""
Analyzes temperature data across multiple datasets.
"""

from typing import Optional
import matplotlib.pyplot as plt
from .base_analysis import BaseAnalysis


class TemperatureAnalysis(BaseAnalysis):
    """
    Analyzes temperature data across multiple datasets.

    This analysis class coordinates temperature analysis across baseline, ullage, and liquid datasets.
    It uses the liquid level processor to determine run start/end times and then analyzes
    temperature data within those time windows.
    """

    def __init__(self, dataset_manager):
        """
        Initialize the temperature analysis.

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

    def analyze(self, show: bool = False, ax: Optional[plt.Axes] = None,
                manual: bool = False, integration_time_ini: int = 60,
                integration_time_end: int = 480, offset_ini: int = 60,
                offset_end: int = 0) -> 'TemperatureAnalysis':
        """
        Analyze temperature across all datasets.

        Args:
            show: Whether to display the plot (not used in current implementation)
            ax: Matplotlib axes to plot on (creates new one if None)
            manual: Whether to use manual time selection
            integration_time_ini: Initial integration time in minutes
            integration_time_end: Final integration time in minutes
            offset_ini: Initial offset in minutes
            offset_end: Final offset in minutes

        Returns:
            Self for method chaining
        """
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
                        start_time, end_time,
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

        # Only add legend if there are labeled artists
        if ax.get_legend_handles_labels()[0]:
            if plotted_datasets:
                ax.legend(ncol=len(plotted_datasets))
            else:
                ax.legend(ncol=1)

        return self
