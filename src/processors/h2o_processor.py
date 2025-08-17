"""
Processes H2O concentration data and provides analysis methods.
"""

from typing import Any, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
from core.base_classes import BaseDataProcessor


class H2OConcentrationProcessor(BaseDataProcessor):
    """
    Processes H2O concentration data and provides analysis methods.

    This processor handles H2O concentration signals and provides methods to:
    - Calculate initial and final H2O concentration values
    - Analyze concentration changes during experiments
    - Plot H2O concentration data with integration windows
    """

    def _validate_data(self):
        """Validate that H2O concentration data is available."""
        if self.standard_data.h2o_concentration is None:
            self.has_h2o_data = False
        else:
            self.has_h2o_data = True

    def process(self, **kwargs) -> Any:
        """
        Process the H2O concentration data and return results.

        For H2O concentration, we typically want to calculate statistics.
        """
        return self.calculate_h2o_concentration(**kwargs)

    def calculate_h2o_concentration(self, start_time, end_time, integration_time_ini=60, integration_time_end=480, offset_ini=60, offset_end=0):
        """
        Calculate H2O concentration analysis results.

        Args:
            start_time: Start time of the run
            end_time: End time of the run
            integration_time_ini: Initial integration time in minutes (time before t0=0)
            integration_time_end: Final integration time in minutes (time ending at offset_end before end)
            offset_ini: Initial offset in minutes (not used in new logic)
            offset_end: Final offset in minutes (time before end where final integration ends)

        Returns:
            Dictionary with analysis results
        """
        if not self.has_h2o_data:
            return None

        try:
            # Get H2O concentration data
            h2o_data = self.standard_data.h2o_concentration
            timestamp = h2o_data.index  # Use the data's own index

            # Calculate initial integration window: integration_time_ini minutes BEFORE t0=0 (start_time)
            ini_start = start_time - pd.Timedelta(minutes=integration_time_ini)
            ini_end = start_time  # t0=0 (run start)

            # Calculate final integration window: integration_time_end minutes ENDING at offset_end minutes BEFORE end_time
            end_end = end_time - pd.Timedelta(minutes=offset_end)  # offset_end minutes before end
            end_start = end_end - pd.Timedelta(minutes=integration_time_end)  # integration_time_end minutes before that

            # Get data for initial calculation (before run starts)
            pre_run_mask = (timestamp >= ini_start) & (timestamp <= ini_end)
            pre_run_data = h2o_data[pre_run_mask]

            # Get data for final calculation (ending before run ends)
            end_mask = (timestamp >= end_start) & (timestamp <= end_end)
            end_data = h2o_data[end_mask]

            # Calculate initial values from pre-run data
            if len(pre_run_data) > 0:
                h2o_ini = pre_run_data.mean()
                h2o_ini_err = pre_run_data.std()
            else:
                # Fallback: use first few points if pre-run data is insufficient
                fallback_data = h2o_data.head(min(10, len(h2o_data)))
                h2o_ini = fallback_data.mean()
                h2o_ini_err = fallback_data.std()

            # Calculate final values from end data
            if len(end_data) > 0:
                h2o_end = end_data.mean()
                h2o_end_err = end_data.std()
            else:
                # Fallback: use last few points if end data is insufficient
                fallback_data = h2o_data.tail(min(10, len(h2o_data)))
                h2o_end = fallback_data.mean()
                h2o_end_err = fallback_data.std()

            # Get run data (from start to end)
            run_mask = (timestamp >= start_time) & (timestamp <= end_time)
            run_data = h2o_data[run_mask]
            run_timestamp = timestamp[run_mask]

            return {
                'h2o_ini': h2o_ini,
                'h2o_ini_err': h2o_ini_err,
                'h2o_end': h2o_end,
                'h2o_end_err': h2o_end_err,
                'full_data': h2o_data,  # All data including pre-run
                'full_timestamp': timestamp,  # All timestamps including pre-run
                'run_data': run_data,  # Data during the run
                'run_timestamp': run_timestamp,  # Timestamps during the run
                'ini_window': (ini_start, ini_end),  # Initial integration window
                'end_window': (end_start, end_end)   # Final integration window
            }

        except Exception as e:
            print(f"Error calculating H2O concentration: {e}")
            return None

    def plot_h2o_concentration(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                               ax: Optional[plt.Axes] = None, color: Optional[str] = None,
                               integration_time: int = 60, dataset_name: str = None) -> plt.Axes:
        """
        Plot H2O concentration over time.

        Args:
            start_time: Start time of the experimental run
            end_time: End time of the experimental run
            ax: Matplotlib axes to plot on (creates new one if None)
            color: Color for the plot line
            integration_time: Integration time window (not used in current implementation)
            dataset_name: Name of the dataset for legend labeling

        Returns:
            Matplotlib axes object with the plot
        """
        if ax is None:
            fig, ax = plt.subplots()

        if not hasattr(self, 'has_h2o_data') or not self.has_h2o_data:
            return ax

        # Get H2O concentration data
        h2o_data = self.standard_data.h2o_concentration
        if h2o_data is None or h2o_data.empty:
            return ax

        # Use dataset name in legend
        label = f'{dataset_name}' if dataset_name else 'H₂O Concentration'

        # Plot data with full range including pre-run
        # Time axis: negative values for pre-run, 0 for start, positive for run
        time_seconds = (h2o_data.index - start_time).total_seconds()

        ax.plot(time_seconds, h2o_data, "-o",
                label=label, markersize=5.0, color=color)

        # Add vertical line at start time (t=0)
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)

        ax.set_title(r'H$_2$O Concentration Over Time')
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel(r'H$_2$O Concentration [ppb]')
        ax.legend()
        ax.grid()

        return ax
