"""
Processes purity/lifetime data and provides analysis methods.
"""

from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.base_classes import BaseDataProcessor


class PurityProcessor(BaseDataProcessor):
    """
    Processes purity/lifetime data and provides analysis methods.

    This processor handles purity/lifetime signals and provides methods to:
    - Calculate initial and final purity values
    - Analyze purity changes during experiments
    - Plot purity data with optional curve fitting
    """

    def _validate_data(self):
        """Validate that purity data is available."""
        if self.standard_data.purity is None:
            self.has_purity_data = False
        else:
            self.has_purity_data = True

    def process(self, **kwargs) -> Any:
        """
        Process the purity data and return results.

        For purity, we typically want to calculate statistics and fit.
        """
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
        """
        Plot purity over time with integration windows.

        Args:
            start_time: Start time of the experimental run
            end_time: End time of the experimental run
            purity_data: Dictionary containing purity analysis results
            ax: Matplotlib axes to plot on (creates new one if None)
            color: Color for the plot line
            fit_legend: Whether to show fit legend
            dataset_name: Name of the dataset for legend labeling

        Returns:
            Matplotlib axes object with the plot
        """
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
