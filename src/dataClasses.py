import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
from typing import Optional, Dict, Any, Tuple

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

    def __init__(self, data: pd.DataFrame, datetime_mapping: Dict[str, str]):
        self.data = data
        self.datetime_mapping = datetime_mapping

    @abstractmethod
    def process(self, **kwargs) -> Any:
        """Process the data and return results."""
        pass


class LiquidLevelProcessor(BaseDataProcessor):
    """Processes liquid level data and provides analysis methods."""

    def __init__(self, data: pd.DataFrame, datetime_mapping: Dict[str, str]):
        super().__init__(data, datetime_mapping)
        self._validate_liquid_level_data()

    def process(self, **kwargs) -> Any:
        """Process the liquid level data and return results."""
        # This method is required by the abstract base class
        # For liquid level, we typically want to find times
        return self.find_times(**kwargs)

    def _validate_liquid_level_data(self):
        """Validate that liquid level data is available."""
        if "PAB_S1_LT_13_AR_REAL_F_CV" not in self.data.columns:
            # Instead of raising an error, just warn and set a flag
            self.has_liquid_level = False
        else:
            self.has_liquid_level = True

    def find_times(self, scan_time: int = 10, threshold: float = 0.995,
                   manual: bool = False) -> Tuple[float, pd.Timestamp, float, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """
        Find key time points in liquid level data.

        Returns:
            Tuple of (max_level, max_time, min_level, min_time, start_time, end_time)
        """
        if not hasattr(self, 'has_liquid_level') or not self.has_liquid_level:
            # Return default times if no liquid level data
            # Use the first and last available datetime from any column
            datetime_cols = [col for col in self.data.columns if "Date-Time" in col]
            if datetime_cols:
                first_col = datetime_cols[0]
                start_time = self.data[first_col].min()
                end_time = self.data[first_col].max()
                return 0.0, start_time, 0.0, start_time, start_time, end_time
            else:
                raise ValueError("No datetime columns found for time analysis")

        finding_time = datetime.timedelta(minutes=scan_time)
        dt_col = self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]

        # Find maximum liquid level and time
        max_level = self.data["PAB_S1_LT_13_AR_REAL_F_CV"].max()
        max_time = self.data.loc[
            self.data["PAB_S1_LT_13_AR_REAL_F_CV"] == max_level
        ][dt_col].iloc[0]

        # Find minimum liquid level within time window
        min_mask = (
            (self.data[dt_col] > max_time) &
            (self.data[dt_col] < max_time + finding_time)
        )
        min_level = self.data["PAB_S1_LT_13_AR_REAL_F_CV"].loc[min_mask].min()
        min_time = self.data.loc[
            (self.data[dt_col] > max_time) &
            (self.data["PAB_S1_LT_13_AR_REAL_F_CV"] == min_level)
        ].index[0]

        # Find start of significant drop
        drop_mask = self.data["PAB_S1_LT_13_AR_REAL_F_CV"] < threshold * min_level
        start_time = self.data.loc[drop_mask][dt_col].iloc[0]

        if manual:
            start_time = input(f"Enter start time (default: {start_time}): ")
            start_time = pd.to_datetime(start_time)

        end_time = start_time + datetime.timedelta(days=1)
        if end_time > self.data[dt_col].max():
            end_time = self.data[dt_col].max()

        return max_level, max_time, min_level, min_time, start_time, end_time

    def plot_level(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                   ax: Optional[plt.Axes] = None, color: Optional[str] = None,
                   integration_time: int = 60, fit_legend: bool = False, dataset_name: str = None) -> plt.Axes:
        """Plot liquid level over time."""
        if ax is None:
            fig, ax = plt.subplots()

        if not hasattr(self, 'has_liquid_level') or not self.has_liquid_level:
            return ax

        dt_col = self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]
        level_data = self.data.set_index(dt_col)["PAB_S1_LT_13_AR_REAL_F_CV"]

        # Calculate initial and final levels
        level_ini = level_data.loc[
            (level_data.index > start_time) &
            (level_data.index < start_time + datetime.timedelta(minutes=integration_time))
        ].mean()
        level_ini_err = level_data.loc[
            (level_data.index > start_time) &
            (level_data.index < start_time + datetime.timedelta(minutes=integration_time))
        ].std()

        level_end = level_data.loc[
            (level_data.index > end_time - datetime.timedelta(minutes=integration_time)) &
            (level_data.index < end_time)
        ].mean()
        level_end_err = level_data.loc[
            (level_data.index > end_time - datetime.timedelta(minutes=integration_time)) &
            (level_data.index < end_time)
        ].std()

        # Plot data with dataset name in legend
        if fit_legend:
            label = f'{dataset_name}: Initial {level_ini:.2f} ± {level_ini_err:.2f} in; Final {level_end:.2f} ± {level_end_err:.2f} in'
        else:
            label = f'{dataset_name}' if dataset_name else 'Liquid Level'

        ax.plot((level_data.index - start_time).total_seconds(), level_data,
                label=label, color=color)

        ax.set_xlabel("Time since start [s]")
        ax.set_ylabel("Liquid Level [in]")
        ax.set_title("Liquid Level Over Time")
        ax.grid()
        ax.legend()

        return ax


class H2OConcentrationProcessor(BaseDataProcessor):
    """Processes H2O concentration data."""

    def __init__(self, data: pd.DataFrame, datetime_mapping: Dict[str, str]):
        super().__init__(data, datetime_mapping)
        self._validate_h2o_data()

    def process(self, **kwargs) -> Any:
        """Process the H2O concentration data and return results."""
        # This method is required by the abstract base class
        # For H2O concentration, we typically want to calculate concentration
        return self.calculate_concentration(**kwargs)

    def _validate_h2o_data(self):
        """Validate that H2O concentration data is available."""
        if "PAB_S1_AE_611_AR_REAL_F_CV" not in self.data.columns:
            self.has_h2o_data = False
        else:
            self.has_h2o_data = True

    def calculate_concentration(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                              integration_time_ini: int = 60, integration_time_end: int = 180,
                              offset_ini: int = 60, offset_end: int = 480) -> Dict[str, float]:
        """Calculate H2O concentration statistics."""
        if not hasattr(self, 'has_h2o_data') or not self.has_h2o_data:
            raise ValueError("Cannot calculate H2O concentration - no H2O data available")

        dt_col = self.datetime_mapping["PAB_S1_AE_611_AR_REAL_F_CV"]
        signal_name = "PAB_S1_AE_611_AR_REAL_F_CV"

        mask = (
            (self.data[dt_col] > start_time - datetime.timedelta(minutes=offset_ini)) &
            (self.data[dt_col] < end_time)
        )
        h2o_data = self.data.loc[mask, [dt_col, signal_name]].set_index(dt_col)[signal_name]

        # Calculate initial concentration
        h2o_ini = h2o_data.loc[
            (h2o_data.index > start_time - datetime.timedelta(minutes=integration_time_ini)) &
            (h2o_data.index < start_time)
        ].mean()
        h2o_ini_err = h2o_data.loc[
            (h2o_data.index > start_time - datetime.timedelta(minutes=integration_time_ini)) &
            (h2o_data.index < start_time)
        ].std()

        # Calculate final concentration
        h2o_end = h2o_data.loc[
            (h2o_data.index > end_time - datetime.timedelta(minutes=offset_end + integration_time_end)) &
            (h2o_data.index < end_time - datetime.timedelta(minutes=offset_end))
        ].mean()
        h2o_end_err = h2o_data.loc[
            (h2o_data.index > end_time - datetime.timedelta(minutes=offset_end + integration_time_end)) &
            (h2o_data.index < end_time - datetime.timedelta(minutes=offset_end))
        ].std()

        return {
            'initial': h2o_ini,
            'initial_error': h2o_ini_err,
            'final': h2o_end,
            'final_error': h2o_end_err,
            'data': h2o_data
        }

    def plot_concentration(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                          concentration_data: Dict[str, Any], ax: Optional[plt.Axes] = None,
                          color: Optional[str] = None, dataset_name: str = None) -> plt.Axes:
        """Plot H2O concentration over time."""
        if ax is None:
            fig, ax = plt.subplots()

        h2o_data = concentration_data['data']
        h2o_ini = concentration_data['initial']
        h2o_ini_err = concentration_data['initial_error']
        h2o_end = concentration_data['final']
        h2o_end_err = concentration_data['final_error']

        # Use dataset name in legend for the main data line only
        label = f'{dataset_name}' if dataset_name else 'H2O Concentration'

        ax.plot((h2o_data.index - start_time).total_seconds(), h2o_data,
                label=label, color=color)
        ax.axhline(y=h2o_ini, color=f'dark{color}', linestyle='--',
                   label=f'Initial: {h2o_ini:.1f} ± {h2o_ini_err:.1f} ppb')
        ax.axhline(y=h2o_end, color=color, linestyle='--',
                   label=f'Final: {h2o_end:.1f} ± {h2o_end_err:.1f} ppb')

        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel(r'H$_2$O Concentration [ppb]')
        ax.set_title(r'H$_2$O Concentration Over Time')
        ax.legend()
        ax.grid()

        return ax


class TemperatureProcessor(BaseDataProcessor):
    """Processes temperature data."""

    def __init__(self, data: pd.DataFrame, datetime_mapping: Dict[str, str]):
        super().__init__(data, datetime_mapping)
        self._validate_temperature_data()

    def process(self, **kwargs) -> Any:
        """Process the temperature data and return results."""
        # This method is required by the abstract base class
        # For temperature, we typically want to calculate temperature
        return self.calculate_temperature(**kwargs)

    def _validate_temperature_data(self):
        """Validate that temperature data is available."""
        if "PAB_S1_TE_324_AR_REAL_F_CV" not in self.data.columns:
            self.has_temperature_data = False
        else:
            self.has_temperature_data = True

    def calculate_temperature(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                            offset_ini: int = 60, integration_time_ini: int = 60) -> Dict[str, Any]:
        """Calculate temperature statistics."""
        if not hasattr(self, 'has_temperature_data') or not self.has_temperature_data:
            raise ValueError("Cannot calculate temperature - no temperature data available")

        dt_col = self.datetime_mapping["PAB_S1_TE_324_AR_REAL_F_CV"]
        signal_name = "PAB_S1_TE_324_AR_REAL_F_CV"

        mask = (
            (self.data[dt_col] > start_time - datetime.timedelta(minutes=offset_ini)) &
            (self.data[dt_col] < end_time)
        )
        temp_data = self.data.loc[mask, [dt_col, signal_name]].set_index(dt_col)[signal_name]

        if temp_data.empty:
            raise ValueError("Temperature data is empty.")

        temp_data = temp_data.dropna()
        if temp_data.empty:
            raise ValueError("Temperature data contains only NaN values.")

        # Calculate initial temperature
        temp_ini = temp_data.loc[
            (temp_data.index > start_time) &
            (temp_data.index < start_time + datetime.timedelta(minutes=integration_time_ini))
        ].mean()
        temp_ini_err = temp_data.loc[
            (temp_data.index > start_time) &
            (temp_data.index < start_time + datetime.timedelta(minutes=integration_time_ini))
        ].std()

        # Calculate final temperature
        temp_end = temp_data.loc[
            (temp_data.index > end_time - datetime.timedelta(minutes=integration_time_ini)) &
            (temp_data.index < end_time)
        ].mean()
        temp_end_err = temp_data.loc[
            (temp_data.index > end_time - datetime.timedelta(minutes=integration_time_ini)) &
            (temp_data.index < end_time)
        ].std()

        return {
            'data': temp_data,
            'initial': temp_ini,
            'initial_error': temp_ini_err,
            'final': temp_end,
            'final_error': temp_end_err
        }

    def plot_temperature(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                        temperature_data: Dict[str, Any], ax: Optional[plt.Axes] = None,
                        color: Optional[str] = None, dataset_name: str = None) -> plt.Axes:
        """Plot temperature over time."""
        if ax is None:
            fig, ax = plt.subplots()

        temp_data = temperature_data['data']
        temp_ini = temperature_data['initial']
        temp_ini_err = temperature_data['initial_error']
        temp_end = temperature_data['final']
        temp_end_err = temperature_data['final_error']

        # Use dataset name in legend for the main data line only
        label = f'{dataset_name}' if dataset_name else 'Temperature'

        ax.plot((temp_data.index - start_time).total_seconds(), temp_data,
                label=label, color=color)
        ax.axhline(y=temp_ini, color=color, linestyle='--',
                   label=f'Initial: {temp_ini:.1f} ± {temp_ini_err:.1f} K')
        ax.axhline(y=temp_end, color=color, linestyle='--',
                   label=f'Final: {temp_end:.1f} ± {temp_end_err:.1f} K')

        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel('Temperature [K]')
        ax.set_title('Sample Temperature Over Time')
        ax.legend()
        ax.grid()

        return ax


class PurityProcessor(BaseDataProcessor):
    """Processes electron lifetime (purity) data."""

    def __init__(self, data: pd.DataFrame, datetime_mapping: Dict[str, str]):
        super().__init__(data, datetime_mapping)
        self._validate_purity_data()

    def process(self, **kwargs) -> Any:
        """Process the purity data and return results."""
        # This method is required by the abstract base class
        # For purity, we typically want to calculate purity
        return self.calculate_purity(**kwargs)

    def exp(self, x, A, tau, C):
        """Exponential decay function: A*exp(-x/tau) + C"""
        return A * np.exp(-x / tau) + C

    def _validate_purity_data(self):
        """Validate that purity data is available."""
        if "Luke_PRM_LIFETIME_F_CV" not in self.data.columns:
            self.has_purity_data = False
        else:
            self.has_purity_data = True

    def calculate_purity(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                        offset_ini: int = 60) -> Dict[str, Any]:
        """Calculate purity statistics and fit exponential decay."""
        if not hasattr(self, 'has_purity_data') or not self.has_purity_data:
            raise ValueError("Cannot calculate purity - no purity data available")

        dt_col = self.datetime_mapping["Luke_PRM_LIFETIME_F_CV"]
        signal_name = "Luke_PRM_LIFETIME_F_CV"

        mask = (
            (self.data[dt_col] > start_time - datetime.timedelta(minutes=offset_ini)) &
            (self.data[dt_col] < end_time)
        )
        purity_data = self.data.loc[mask, [dt_col, signal_name]].set_index(dt_col)[signal_name]

        if purity_data.empty:
            raise ValueError("Purity data is empty.")

        purity_data = purity_data.dropna()
        if purity_data.empty:
            raise ValueError("Purity data contains only NaN values.")

        # Fit exponential curve
        time_seconds = (purity_data.index - purity_data.index[0]).total_seconds()
        popt, pcov = curve_fit(
            self.exp, time_seconds, purity_data.values,
            p0=[purity_data.iloc[0], 8*3600, purity_data.iloc[-1]]
        )

        return {
            'data': purity_data,
            'fit_parameters': popt,
            'fit_covariance': pcov,
            'fit_errors': np.sqrt(np.diag(pcov))
        }

    def plot_purity(self, start_time: pd.Timestamp, end_time: pd.Timestamp,
                    purity_data: Dict[str, Any], ax: Optional[plt.Axes] = None,
                    color: Optional[str] = None, fit_legend: bool = False, dataset_name: str = None) -> plt.Axes:
        """Plot purity over time with optional fit curve."""
        if ax is None:
            fig, ax = plt.subplots()

        data = purity_data['data']
        popt = purity_data['fit_parameters']
        pcov = purity_data['fit_covariance']

        # Use dataset name in legend
        label = f'{dataset_name}' if dataset_name else 'Purity'

        # Plot data
        ax.plot((data.index - start_time).total_seconds(), data, "-o",
                label=label, markersize=5.0, color=color)

        # Plot fit if requested
        if fit_legend:
            time_seconds = (data.index - data.index[0]).total_seconds()
            fit_curve = self.exp(time_seconds, *popt)
            fit_label = f'{dataset_name} Fit: A={1e3*popt[0]:.2f}±{1e3*np.sqrt(pcov[0,0]):.2f} ms, τ={(1/3600)*popt[1]:.2f}±{(1/3600)*np.sqrt(pcov[1,1]):.2f} h, C={1e3*popt[2]:.2f}±{1e3*np.sqrt(pcov[2,2]):.2f} ms' if dataset_name else f'Fit: A={1e3*popt[0]:.2f}±{1e3*np.sqrt(pcov[0,0]):.2f} ms, τ={(1/3600)*popt[1]:.2f}±{(1/3600)*np.sqrt(pcov[1,1]):.2f} h, C={1e3*popt[2]:.2f}±{1e3*np.sqrt(pcov[2,2]):.2f} ms'
            ax.plot((data.index - start_time).total_seconds(), fit_curve,
                    label=fit_label, linestyle='--', color=color)
            ax.set_title(r'$e^-$ lifetime Over Time - Fit: $Ae^{\frac{x}{\tau}} + C$')
        else:
            ax.set_title(r'$e^-$ lifetime Over Time')

        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel(r'$e^-$ lifetime [s]')
        ax.legend()
        ax.grid()

        return ax


class Dataset:
    """
    A class to handle datasets stored in Excel files, providing methods to load, describe, and visualize the data.

    This class has been restructured to use specialized processors for different data types,
    making it more modular and maintainable.
    """

    def __init__(self, path: str, name: Optional[str] = None):
        self.path = path
        self.name = name or DataValidator.validate_file_path(path)
        self.sheet_names = self._get_sheet_names()

        # Data storage
        self.configuration: Optional[pd.DataFrame] = None
        self.info: Optional[pd.DataFrame] = None
        self.data: Optional[pd.DataFrame] = None
        self.datetime_mapping: Dict[str, str] = {}
        self.unique_datetime: bool = False

        # Processors
        self.liquid_level: Optional[LiquidLevelProcessor] = None
        self.h2o_concentration: Optional[H2OConcentrationProcessor] = None
        self.temperature: Optional[TemperatureProcessor] = None
        self.purity: Optional[PurityProcessor] = None

        # Analysis results
        self._analysis_results: Dict[str, Any] = {}

    def _get_sheet_names(self) -> list:
        """Get the names of the sheets in the Excel file."""
        try:
            return pd.ExcelFile(self.path).sheet_names
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

    def load(self) -> 'Dataset':
        """Load the dataset from the specified path."""
        try:
            for sheet_name in self.sheet_names:
                if "Info" in sheet_name:
                    self.configuration = pd.read_excel(self.path, sheet_name=sheet_name)
                elif "Signal" in sheet_name:
                    self.info = pd.read_excel(self.path, sheet_name=sheet_name, header=0)
                elif "Grid" in sheet_name or "Data" in sheet_name:
                    self.data = pd.read_excel(self.path, sheet_name=sheet_name, header=0)

            # Initialize processors after loading data and assigning datetime
            if self.data is not None:
                self.assign_datetime()
                self._initialize_processors()

        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data = None
            self.configuration = None
            self.info = None

        return self

    def _initialize_processors(self):
        """Initialize data processors after data is loaded."""
        if self.data is not None and not self.data.empty and hasattr(self, 'datetime_mapping'):
            try:
                self.liquid_level = LiquidLevelProcessor(self.data, self.datetime_mapping)
                self.h2o_concentration = H2OConcentrationProcessor(self.data, self.datetime_mapping)
                self.temperature = TemperatureProcessor(self.data, self.datetime_mapping)
                self.purity = PurityProcessor(self.data, self.datetime_mapping)
            except Exception as e:
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

    def assign_datetime(self) -> 'Dataset':
        """Assign datetime mapping for data columns."""
        self.datetime_mapping = {}

        for index, name in enumerate(self.data.columns):
            if "Date-Time" in name:
                if index + 1 < len(self.data.columns):
                    self.datetime_mapping[self.data.columns[index + 1]] = name
                self.data[name] = pd.to_datetime(self.data[name], errors='coerce')

        if len(self.datetime_mapping) == 1:
            for name in self.data.columns:
                if "Date-Time" not in name:
                    self.datetime_mapping[name] = "Date-Time"
            self.unique_datetime = True
        elif len(self.datetime_mapping) > 1:
            self.unique_datetime = False
        else:
            self.unique_datetime = False

        return self

    def describe(self) -> None:
        """Print a description of the dataset."""
        if self.info is not None:
            print(self.info)
        else:
            print("No info available. Please load the dataset first.")

    def show(self) -> None:
        """Plot the data in the dataset."""
        if self.data is None or self.data.empty:
            print("No data to show. Please load the dataset first.")
            return

        if self.info is None or self.info.empty:
            print("No info to show. Please load the dataset first.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        fig.suptitle(f"{self.name} Dataset", fontsize=16, fontweight='bold')
        axes = axes.flatten()

        cnt = 0
        for signal in self.data.columns:
            if "Date-Time" in signal or cnt >= 6:
                continue

            dt_col = self.datetime_mapping.get(signal, "Date-Time")
            axes[cnt].plot(self.data[dt_col], self.data[signal], "o")
            axes[cnt].tick_params(axis='x', rotation=45)

            # Set labels based on signal type
            if "PRM" in signal:
                axes[cnt].set_title(r"$e^-$-lifetime")
                axes[cnt].set_ylabel(r"$e^-$-lifetime [s]")
            elif "AE" in signal:
                axes[cnt].set_title(r"$H_2$O Concentration")
                axes[cnt].set_ylabel(r"$H_2$O Concentration [ppb]")
            elif "TE" in signal:
                axes[cnt].set_title("Temperature")
                axes[cnt].set_ylabel("Temperature [K]")
            elif "LT" in signal:
                axes[cnt].set_title("Liquid Level")
                axes[cnt].set_ylabel("Liquid Level [in]")
            elif "PRESSURE" in signal:
                axes[cnt].set_title("Pressure")
                axes[cnt].set_ylabel("Pressure [PSIG]")

            if cnt > 1:
                axes[cnt].set_xlabel("Datetime")
            axes[cnt].grid()
            cnt += 1

        fig.tight_layout()
        plt.show()

    # Legacy method names for backward compatibility
    def findTimes(self, **kwargs):
        """Legacy method - use liquid_level.find_times() instead."""
        if self.liquid_level is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        return self.liquid_level.find_times(**kwargs)

    def level(self, **kwargs):
        """Legacy method - use liquid_level.plot_level() instead."""
        if self.liquid_level is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        return self.liquid_level.plot_level(**kwargs)

    def h2oConcentration(self, **kwargs):
        """Legacy method - use h2o_concentration methods instead."""
        if self.h2o_concentration is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        return self.h2o_concentration.calculate_concentration(**kwargs)

    def temperature(self, **kwargs):
        """Legacy method - use temperature methods instead."""
        if self.temperature is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        return self.temperature.calculate_temperature(**kwargs)

    def purity(self, **kwargs):
        """Legacy method - use purity methods instead."""
        if self.purity is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        return self.purity.calculate_purity(**kwargs)