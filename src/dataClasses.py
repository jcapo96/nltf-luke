import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
from typing import Optional, Dict, Any, Tuple

try:
    from .dataFormats import StandardDataFormat
except ImportError:
    # Fallback for when running as script
    from dataFormats import StandardDataFormat

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="openpyxl.styles.stylesheet",
    message="Workbook contains no default style, apply openpyxl's default"
)


class DataValidator:
    """Utility class for data validation operations."""

    @staticmethod
    def validate_file_path(path: str) -> str:
        """Validate and return the dataset type from file path."""
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string.")
        if not path.endswith('.xlsx'):
            raise ValueError("Path must point to an Excel file with .xlsx extension.")

        if "baseline" in path:
            return "Baseline"
        elif "ullage" in path:
            return "Ullage"
        elif "liquid" in path:
            return "Liquid"
        else:
            return "Unknown"

    @staticmethod
    def validate_dataframe(data: Any, name: str) -> None:
        """Validate that data is a non-empty pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        if data.empty:
            raise ValueError(f"{name} cannot be empty.")


class BaseDataProcessor(ABC):
    """Abstract base class for data processing operations."""

    def __init__(self, standard_data: StandardDataFormat):
        self.standard_data = standard_data
        self._validate_data()

    @abstractmethod
    def process(self, **kwargs) -> Any:
        """Process the data and return results."""
        pass

    @abstractmethod
    def _validate_data(self):
        """Validate that required data is available."""
        pass


class LiquidLevelProcessor(BaseDataProcessor):
    """Processes liquid level data and provides analysis methods."""

    def _validate_data(self):
        """Validate that liquid level data is available."""
        if self.standard_data.liquid_level is None:
            self.has_liquid_level = False
        else:
            self.has_liquid_level = True

    def process(self, **kwargs) -> Any:
        """Process the liquid level data and return results."""
        # This method is required by the abstract base class
        # For liquid level, we typically want to find times
        return self.find_times(**kwargs)

    def find_times(self, manual: bool = False) -> dict:
        """
        Find the start and end times of the experimental run based on liquid level patterns.

        The method detects:
        1. Start: When level drops from initial oscillation pattern to stable pattern
        2. End: When level rises again or returns to oscillation pattern
        3. If detected duration < 12 hours, extend end time to 24 hours after start

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

        # If detected duration is shorter than 12 hours, extend to 24 hours
        if detected_duration_hours < 12:
            target_end_time = start_time + pd.Timedelta(hours=24)
            # But don't exceed the available data
            if target_end_time <= full_end_time:
                end_time = target_end_time
                print(f"Warning: Detected duration ({detected_duration_hours:.1f}h) < 12h, extending to 24h")
            else:
                print(f"Warning: Detected duration ({detected_duration_hours:.1f}h) < 12h, but cannot extend to 24h (data limit)")

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
        """Plot liquid level over time."""
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


class H2OConcentrationProcessor(BaseDataProcessor):
    """Processes H2O concentration data and provides analysis methods."""

    def _validate_data(self):
        """Validate that H2O concentration data is available."""
        if self.standard_data.h2o_concentration is None:
            self.has_h2o_data = False
        else:
            self.has_h2o_data = True

    def process(self, **kwargs) -> Any:
        """Process the H2O concentration data and return results."""
        # For H2O concentration, we typically want to calculate statistics
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

    def plot_concentration(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                          h2o_data: Dict[str, Any], ax: Optional[plt.Axes] = None,
                          color: Optional[str] = None, integration_time: int = 60, dataset_name: str = None) -> plt.Axes:
        """Plot H2O concentration over time."""
        if ax is None:
            fig, ax = plt.subplots()

        if not hasattr(self, 'has_h2o_data') or not self.has_h2o_data:
            return ax

        # Extract data
        plot_data = h2o_data['full_data']  # Full data including pre-run
        plot_timestamp = h2o_data['full_timestamp']  # Full timestamp including pre-run
        h2o_ini = h2o_data['h2o_ini']
        h2o_end = h2o_data['h2o_end']
        h2o_ini_err = h2o_data['h2o_ini_err']
        h2o_end_err = h2o_data['h2o_end_err']

        # Use dataset name in legend
        label = f'{dataset_name}' if dataset_name else 'H₂O Concentration'

        # Plot data with full range including pre-run
        # Time axis: negative values for pre-run, 0 for start, positive for run
        time_seconds = (plot_timestamp - start_time).total_seconds()

        ax.plot(time_seconds, plot_data, "-o",
                label=label, markersize=5.0, color=color)

        # Plot initial and final values
        ax.axhline(y=h2o_ini, color=color, linestyle='--', alpha=0.7,
                   label=f'Initial: {h2o_ini:.1f} ± {h2o_ini_err:.1f} ppb')
        ax.axhline(y=h2o_end, color=color, linestyle='--', alpha=0.7,
                   label=f'Final: {h2o_end:.1f} ± {h2o_end_err:.1f} ppb')

        # Add vertical line at start time (t=0)
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)

        ax.set_title(r'H$_2$O Concentration Over Time')
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel(r'H$_2$O Concentration [ppb]')
        ax.legend()
        ax.grid()

        return ax


class TemperatureProcessor(BaseDataProcessor):
    """Processes temperature data and provides analysis methods."""

    def _validate_data(self):
        """Validate that temperature data is available."""
        if self.standard_data.temperature is None:
            self.has_temperature_data = False
        else:
            self.has_temperature_data = True

    def process(self, **kwargs) -> Any:
        """Process the temperature data and return results."""
        # For temperature, we typically want to calculate statistics
        return self.calculate_temperature(**kwargs)

    def calculate_temperature(self, start_time, end_time, integration_time_ini=60, integration_time_end=480, offset_ini=60, offset_end=0):
        """
        Calculate temperature analysis results.

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
        if not self.has_temperature_data:
            return None

        try:
            # Get temperature data
            temp_data = self.standard_data.temperature
            timestamp = temp_data.index  # Use the data's own index

            # Calculate initial integration window: integration_time_ini minutes BEFORE t0=0 (start_time)
            ini_start = start_time - pd.Timedelta(minutes=integration_time_ini)
            ini_end = start_time  # t0=0 (run start)

            # Calculate final integration window: integration_time_end minutes ENDING at offset_end minutes BEFORE end_time
            end_end = end_time - pd.Timedelta(minutes=offset_end)  # offset_end minutes before end
            end_start = end_end - pd.Timedelta(minutes=integration_time_end)  # integration_time_end minutes before that

            # Get data for initial calculation (before run starts)
            pre_run_mask = (timestamp >= ini_start) & (timestamp <= ini_end)
            pre_run_data = temp_data[pre_run_mask]

            # Get data for final calculation (ending before run ends)
            end_mask = (timestamp >= end_start) & (timestamp <= end_end)
            end_data = temp_data[end_mask]

            # Calculate initial values from pre-run data
            if len(pre_run_data) > 0:
                temp_ini = pre_run_data.mean()
                temp_ini_err = pre_run_data.std()
            else:
                # Fallback: use first few points if pre-run data is insufficient
                fallback_data = temp_data.head(min(10, len(temp_data)))
                temp_ini = fallback_data.mean()
                temp_ini_err = fallback_data.std()

            # Calculate final values from end data
            if len(end_data) > 0:
                temp_end = end_data.mean()
                temp_end_err = end_data.std()
            else:
                # Fallback: use last few points if end data is insufficient
                fallback_data = temp_data.tail(min(10, len(temp_data)))
                temp_end = fallback_data.mean()
                temp_end_err = fallback_data.std()

            # Get run data (from start to end)
            run_mask = (timestamp >= start_time) & (timestamp <= end_time)
            run_data = temp_data[run_mask]
            run_timestamp = timestamp[run_mask]

            return {
                'temp_ini': temp_ini,
                'temp_ini_err': temp_ini_err,
                'temp_end': temp_end,
                'temp_end_err': temp_end_err,
                'full_data': temp_data,  # All data including pre-run
                'full_timestamp': timestamp,  # All timestamps including pre-run
                'run_data': run_data,  # Data during the run
                'run_timestamp': run_timestamp,  # Timestamps during the run
                'ini_window': (ini_start, ini_end),  # Initial integration window
                'end_window': (end_start, end_end)   # Final integration window
            }

        except Exception as e:
            print(f"Error calculating temperature: {e}")
            return None

    def plot_temperature(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                        temp_data: Dict[str, Any], ax: Optional[plt.Axes] = None,
                        color: Optional[str] = None, integration_time: int = 60, dataset_name: str = None) -> plt.Axes:
        """Plot temperature over time."""
        if ax is None:
            fig, ax = plt.subplots()

        if not hasattr(self, 'has_temperature_data') or not self.has_temperature_data:
            return ax

        # Extract data
        plot_data = temp_data['full_data']  # Full data including pre-run
        plot_timestamp = temp_data['full_timestamp']  # Full timestamp including pre-run
        temp_ini = temp_data['temp_ini']
        temp_end = temp_data['temp_end']
        temp_ini_err = temp_data['temp_ini_err']
        temp_end_err = temp_data['temp_end_err']

        # Use dataset name in legend
        label = f'{dataset_name}' if dataset_name else 'Temperature'

        # Plot data with full range including pre-run
        # Time axis: negative values for pre-run, 0 for start, positive for run
        time_seconds = (plot_timestamp - start_time).total_seconds()

        ax.plot(time_seconds, plot_data, "-o",
                label=label, markersize=5.0, color=color)

        # Plot initial and final values
        ax.axhline(y=temp_ini, color=color, linestyle='--', alpha=0.7,
                   label=f'Initial: {temp_ini:.2f} ± {temp_ini_err:.2f} K')
        ax.axhline(y=temp_end, color=color, linestyle='--', alpha=0.7,
                   label=f'Final: {temp_end:.2f} ± {temp_end_err:.2f} K')

        # Add vertical line at start time (t=0)
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5, label='Run Start')

        ax.set_title('Temperature Over Time')
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel('Temperature [K]')
        ax.legend()
        ax.grid()

        return ax


class PurityProcessor(BaseDataProcessor):
    """Processes purity/lifetime data and provides analysis methods."""

    def _validate_data(self):
        """Validate that purity data is available."""
        if self.standard_data.purity is None:
            self.has_purity_data = False
        else:
            self.has_purity_data = True

    def process(self, **kwargs) -> Any:
        """Process the purity data and return results."""
        # For purity, we typically want to calculate statistics and fit
        return self.calculate_purity(**kwargs)

    def exp(self, x, A, tau, C):
        """Exponential decay function for curve fitting."""
        return A * np.exp(-x / tau) + C

    def calculate_purity(self, start_time, end_time, integration_time_ini=60, integration_time_end=480, offset_ini=60, offset_end=0):
        """
        Calculate purity analysis results.

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
        if not self.has_purity_data:
            return None

        try:
            # Get purity data
            purity_data = self.standard_data.purity
            timestamp = purity_data.index  # Use the data's own index

            # Calculate initial integration window: integration_time_ini minutes BEFORE t0=0 (start_time)
            ini_start = start_time - pd.Timedelta(minutes=integration_time_ini)
            ini_end = start_time  # t0=0 (run start)

            # Calculate final integration window: integration_time_end minutes ENDING at offset_end minutes BEFORE end_time
            end_end = end_time - pd.Timedelta(minutes=offset_end)  # offset_end minutes before end
            end_start = end_end - pd.Timedelta(minutes=integration_time_end)  # integration_time_end minutes before that

            # Get data for initial calculation (before run starts)
            pre_run_mask = (timestamp >= ini_start) & (timestamp <= ini_end)
            pre_run_data = purity_data[pre_run_mask]

            # Get data for final calculation (ending before run ends)
            end_mask = (timestamp >= end_start) & (timestamp <= end_end)
            end_data = purity_data[end_mask]

            # Calculate initial values from pre-run data
            if len(pre_run_data) > 0:
                purity_ini = pre_run_data.mean()
                purity_ini_err = pre_run_data.std()
            else:
                # Fallback: use first few points if pre-run data is insufficient
                fallback_data = purity_data.head(min(10, len(purity_data)))
                purity_ini = fallback_data.mean()
                purity_ini_err = fallback_data.std()

            # Calculate final values from end data
            if len(end_data) > 0:
                purity_end = end_data.mean()
                purity_end_err = end_data.std()
            else:
                # Fallback: use last few points if end data is insufficient
                fallback_data = purity_data.tail(min(10, len(purity_data)))
                purity_end = fallback_data.mean()
                purity_end_err = fallback_data.std()

            # Get run data (from start to end)
            run_mask = (timestamp >= start_time) & (timestamp <= end_time)
            run_data = purity_data[run_mask]
            run_timestamp = timestamp[run_mask]

            return {
                'purity_ini': purity_ini,
                'purity_ini_err': purity_ini_err,
                'purity_end': purity_end,
                'purity_end_err': purity_end_err,
                'full_data': purity_data,  # All data including pre-run
                'full_timestamp': timestamp,  # All timestamps including pre-run
                'run_data': run_data,  # Data during the run
                'run_timestamp': run_timestamp,  # Timestamps during the run
                'ini_window': (ini_start, ini_end),  # Initial integration window
                'end_window': (end_start, end_end)   # Final integration window
            }

        except Exception as e:
            print(f"Error calculating purity: {e}")
            return None

    def plot_purity(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                    purity_data: Dict[str, Any], ax: Optional[plt.Axes] = None,
                    color: Optional[str] = None, fit_legend: bool = False, dataset_name: str = None) -> plt.Axes:
        """Plot purity over time with integration windows."""
        if ax is None:
            fig, ax = plt.subplots()

        # Use dataset name in legend
        label = f'{dataset_name}' if dataset_name else 'Purity'

        # Get the full data and timestamps
        if 'full_data' in purity_data and 'full_timestamp' in purity_data:
            data = purity_data['full_data']
            timestamp = purity_data['full_timestamp']
        else:
            # Fallback to old structure if available
            if 'data' in purity_data:
                data = purity_data['data']
                timestamp = data.index
            else:
                print("Warning: No purity data available for plotting")
                return ax

        # Plot the full data including pre-run
        time_seconds = (timestamp - start_time).total_seconds()
        ax.plot(time_seconds, data, "-o", label=label, markersize=5.0, color=color)

        # Plot fit if requested and available
        if fit_legend and 'fit_parameters' in purity_data and 'fit_covariance' in purity_data:
            popt = purity_data['fit_parameters']
            pcov = purity_data['fit_covariance']

            time_seconds = (timestamp - timestamp[0]).total_seconds()
            fit_curve = self.exp(time_seconds, *popt)
            fit_label = f'{dataset_name} Fit: A={1e3*popt[0]:.2f}±{1e3*np.sqrt(pcov[0,0]):.2f} ms, τ={(1/3600)*popt[1]:.2f}±{(1/3600)*np.sqrt(pcov[1,1]):.2f} h, C={1e3*popt[2]:.2f}±{1e3*np.sqrt(pcov[2,2]):.2f} ms' if dataset_name else f'Fit: A={1e3*popt[0]:.2f}±{1e3*np.sqrt(pcov[0,0]):.2f} ms, τ={(1/3600)*popt[1]:.2f}±{(1/3600)*np.sqrt(pcov[1,1]):.2f} h, C={1e3*popt[2]:.2f}±{1e3*np.sqrt(pcov[2,2]):.2f} ms'
            ax.plot(time_seconds, fit_curve,
                    label=fit_label, linestyle='--', color=color)

        ax.set_title(r'$e^-$ lifetime Over Time')
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel(r'$e^-$ lifetime [s]')
        ax.legend()
        ax.grid()

        return ax


class Dataset:
    """
    A class to handle datasets using the new data format abstraction layer.

    This class now works with StandardDataFormat instead of hardcoded column names,
    making it format-independent and more flexible.
    """

    def __init__(self, path: str, name: Optional[str] = None):
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
        """Load the dataset from the specified path using the format manager."""
        try:
            if format_manager is None:
                from .dataFormats import DataFormatManager
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
                self.h2o_concentration.plot_concentration(start_time, end_time, h2o_data, ax=ax, dataset_name=self.name)
            elif plot_type == "Temperature" and self.temperature:
                temp_data = self.temperature.calculate_temperature(start_time, end_time)
                self.temperature.plot_temperature(start_time, end_time, temp_data, ax=ax, dataset_name=self.name)
            elif plot_type == "Purity" and self.purity:
                purity_data = self.purity.calculate_purity(start_time, end_time)
                self.purity.plot_purity(start_time, end_time, purity_data, ax=ax, dataset_name=self.name)

        plt.tight_layout()
        plt.show()

    def get_analysis_results(self) -> Dict[str, Any]:
        """Get analysis results for this dataset."""
        return self._analysis_results.copy()

    def set_analysis_results(self, analysis_type: str, results: Dict[str, Any]):
        """Set analysis results for this dataset."""
        self._analysis_results[analysis_type] = results