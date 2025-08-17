"""
Processes liquid level data and provides analysis methods.
"""

from typing import Any, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
from core.base_classes import BaseDataProcessor


class LiquidLevelProcessor(BaseDataProcessor):
    """
    Processes liquid level data and provides analysis methods.

    This processor handles liquid level signals and provides methods to:
    - Find experimental run start/end times based on oscillation patterns
    - Plot liquid level data with proper time scaling
    - Analyze level changes during experiments
    """

    def _validate_data(self):
        """Validate that liquid level data is available."""
        if self.standard_data.liquid_level is None:
            self.has_liquid_level = False
        else:
            self.has_liquid_level = True

    def process(self, **kwargs) -> Any:
        """
        Process the liquid level data and return results.

        This method is required by the abstract base class.
        For liquid level, we typically want to find times.
        """
        return self.find_times(**kwargs)

    def find_times(self, manual: bool = False) -> dict:
        """
        Find the start and end times of the experimental run based on liquid level patterns.

        The method detects:
        1. Start: When level drops from initial oscillation pattern to stable pattern
        2. End: When level rises again or returns to oscillation pattern
        3. If detected duration < 18 hours, extend end time to 24 hours after start

        Args:
            manual: If True, use manual start/end times from data

        Returns:
            Dictionary with start_time, end_time, max_time, min_time
        """
        if not self.has_liquid_level:
            return {
                'start_time': None,
                'end_time': None,
                'max_time': None,
                'min_time': None
            }

        level_data = self.standard_data.liquid_level
        if level_data is None or level_data.empty:
            return {
                'start_time': None,
                'end_time': None,
                'max_time': None,
                'min_time': None
            }

        # Get the full time range
        full_start_time = level_data.index[0]
        full_end_time = level_data.index[-1]

        if manual:
            # Use manual start/end times
            return {
                'start_time': full_start_time,
                'end_time': full_end_time,
                'max_time': level_data.idxmax(),
                'min_time': level_data.idxmin()
            }

        # Analyze oscillation patterns to find run start/end
        # Use sliding windows to detect changes in standard deviation

        # Parameters for detection
        window_size = 100  # Number of points for std calculation
        step_size = 50     # Step size for sliding window
        threshold_factor = 3.0  # Factor above which std dev indicates oscillation

        # Calculate baseline oscillation (first window)
        baseline_std = level_data.head(window_size).std()

        # Find start of run (transition from oscillation to stable)
        run_start_idx = None
        for i in range(0, len(level_data) - window_size, step_size):
            window_data = level_data.iloc[i:i+window_size]
            window_std = window_data.std()

            # If std dev drops significantly, we've found the start
            if window_std < baseline_std / threshold_factor:
                run_start_idx = i
                break

        if run_start_idx is None:
            # Fallback: use first 10% of data as pre-run
            run_start_idx = len(level_data) // 10

        # Find end of run (transition back to oscillation)
        run_end_idx = None

        # Start looking from the start point
        search_start = run_start_idx + window_size

        # Look for return to oscillation pattern
        for i in range(search_start, len(level_data) - window_size, step_size):
            window_data = level_data.iloc[i:i+window_size]
            window_std = window_data.std()

            # If std dev increases significantly, we've found the end
            if window_std > baseline_std / threshold_factor:
                run_end_idx = i + window_size
                break

        if run_end_idx is None:
            # No oscillation detected, use the end of available data
            run_end_idx = len(level_data)

        # Convert indices to timestamps
        start_time = level_data.index[run_start_idx]

        # Handle the case when run_end_idx equals the length of data
        if run_end_idx >= len(level_data):
            end_time = level_data.index[-1]
        else:
            end_time = level_data.index[run_end_idx]

        # Calculate the detected duration
        detected_duration_hours = (end_time - start_time).total_seconds() / 3600

        # If detected duration is shorter than 18 hours, extend to at least 24 hours
        if detected_duration_hours < 18:
            target_end_time = start_time + pd.Timedelta(hours=24)
            # Force extension to 24 hours even if it exceeds available data
            end_time = target_end_time
            print(f"Warning: Detected duration ({detected_duration_hours:.1f}h) < 18h, extending to 24h")
            print(f"Note: Extended end time ({end_time}) may exceed available data boundaries")

        # Ensure we don't exceed data bounds
        if end_time > full_end_time:
            end_time = full_end_time

        # Get max and min times within the run period
        # Use proper slicing to avoid index errors
        if run_end_idx >= len(level_data):
            run_data = level_data[run_start_idx:]
        else:
            run_data = level_data[run_start_idx:run_end_idx]
        max_time = run_data.idxmax()
        min_time = run_data.idxmin()

        return {
            'start_time': start_time,
            'end_time': end_time,
            'max_time': max_time,
            'min_time': min_time
        }

    def plot_level(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                   ax: Optional[plt.Axes] = None, color: Optional[str] = None,
                   integration_time: int = 60, fit_legend: bool = False, dataset_name: str = None) -> plt.Axes:
        """
        Plot liquid level over time.

        Args:
            start_time: Start time of the experimental run
            end_time: End time of the experimental run
            ax: Matplotlib axes to plot on (creates new one if None)
            color: Color for the plot line
            integration_time: Integration time window (not used in current implementation)
            fit_legend: Whether to show fit legend (not used in current implementation)
            dataset_name: Name of the dataset for legend labeling

        Returns:
            Matplotlib axes object with the plot
        """
        if ax is None:
            fig, ax = plt.subplots()

        if not hasattr(self, 'has_liquid_level') or not self.has_liquid_level:
            return ax

        # Get data within time range including pre-run
        # Use the signal's own timestamp for filtering, not the primary timestamp
        if self.standard_data.liquid_level is not None:
            # Include pre-run data (before start_time) and run data (up to end_time)
            mask = (self.standard_data.liquid_level.index <= end_time)
            plot_timestamp = self.standard_data.liquid_level.index[mask]
            plot_level = self.standard_data.liquid_level[mask]
        else:
            return ax

        if plot_level.empty:
            return ax

        # Use dataset name in legend
        label = f'{dataset_name}' if dataset_name else 'Liquid Level'

        # Plot data with full range including pre-run
        # Time axis: negative values for pre-run, 0 for start, positive for run
        time_seconds = (plot_timestamp - start_time).total_seconds()

        ax.plot(time_seconds, plot_level, "-o",
                label=label, markersize=5.0, color=color)

        # Add vertical line at start time (t=0)
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)

        ax.set_title('Liquid Level Over Time')
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel('Liquid Level [%]')
        ax.legend()
        ax.grid()

        return ax
