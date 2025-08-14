import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
from scipy.optimize import curve_fit

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="openpyxl.styles.stylesheet",
    message="Workbook contains no default style, apply openpyxl's default"
)

class Dataset():
    """
    A class to handle datasets stored in Excel files, providing methods to load, describe, and visualize the data.
    Attributes
    ----------
    path : str
        The path to the Excel file containing the dataset.
    name : str
        The name of the dataset, derived from the file path.
    sheetNames : list
        The names of the sheets in the Excel file.
    configuration : pd.DataFrame
        DataFrame containing configuration information from the "Info" sheet.
    info : pd.DataFrame
        DataFrame containing signal information from the "Signal" sheet.
    data : pd.DataFrame
        DataFrame containing the main data from the "Grid" sheet, indexed by datetime.
    Methods
    -------
    __init__(path: str, name=None)
        Initializes the Dataset object with the path to the Excel file and sets the dataset name.
    __name__()
        Sets the name of the dataset based on the file path.
    __sheets__()
        Retrieves the names of the sheets in the Excel file.
    load()
        Loads the dataset from the specified Excel file path, reading the "Info", "Signal", and "Grid" sheets.
    describe()
        Prints a description of the dataset, including the info DataFrame.
    show()
        Plots the data in the dataset, creating subplots for each signal in the data.
    findTimes(scan_time=60, threshold=0.99)
        Finds the maximum and minimum liquid levels and their corresponding times, as well as the start time of a significant drop in liquid level within a specified time window.
    Raises
    ------
    ValueError
        If the path is not set, is invalid, or if required data is missing.
    TypeError
        If the path is not a string or if the data is not a pandas DataFrame.
    """

    def __init__(self, path: str, name=None):
        self.path = path
        self.__name__()
        self.__sheets__()

    @staticmethod
    def exp(x, a, b, c):
        """Exponential function for curve fitting."""
        return a * np.exp(-(1/b)*x) + c

    def __name__(self):
        """Set the name of the dataset based on the path."""
        if not hasattr(self, 'path') or not self.path:
            raise ValueError("Path to the Excel file is not set.")
        if not isinstance(self.path, str):
            raise TypeError("Path must be a string.")
        if not self.path.endswith('.xlsx'):
            raise ValueError("Path must point to an Excel file with .xlsx extension.")
        if not self.path:
            raise ValueError("Path cannot be empty.")
        if "baseline" in self.path:
            self.name = "Baseline"
        elif "ullage" in self.path:
            self.name = "Ullage"
        elif "liquid" in self.path:
            self.name = "Liquid"
        else:
            print("Unknown dataset type")
            self.name = None
        return self

    def __sheets__(self):
        """Get the names of the sheets in the Excel file."""
        if not hasattr(self, 'path') or not self.path:
            raise ValueError("Path to the Excel file is not set.")
        if not isinstance(self.path, str):
            raise TypeError("Path must be a string.")
        if not self.path.endswith('.xlsx'):
            raise ValueError("Path must point to an Excel file with .xlsx extension.")
        try:
            self.sheetNames = pd.ExcelFile(self.path).sheet_names
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")
        return self.sheetNames

    def load(self):
        """Load the dataset from the specified path."""
        try:
            for sheetName in self.sheetNames:
                if "Info" in sheetName:
                    self.configuration = pd.read_excel(self.path, sheet_name=sheetName)
                elif "Signal" in sheetName:
                    self.info = pd.read_excel(self.path, sheet_name=sheetName, header=0)
                elif "Grid" in sheetName or "Data" in sheetName:
                    self.data = pd.read_excel(self.path, sheet_name=sheetName, header=0)
                    # self.data.set_index("Date-Time", inplace=True)
                    # self.data.index = pd.to_datetime(self.data.index)
                else:
                    print(f"Unknown sheet: {sheetName}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data = None
            self.configuration = None
            self.info = None
        return self

    def describe(self):
        """Print a description of the dataset."""
        if hasattr(self, 'info'):
            print(self.info)

    def assign_datetime(self):
        self.datetime_mapping = {}
        for index, name in enumerate(self.data.columns):
            if "Date-Time" in name:
                self.datetime_mapping[self.data.columns[index+1]] = name
                self.data[name] = pd.to_datetime(self.data[name], errors='coerce')
        if len(self.datetime_mapping) == 1:
            print("Only one Date-Time column found. Assigning it to all signals.")
            for index, name in enumerate(self.data.columns):
                if "Date-Time" in name:
                    continue
                else:
                    self.datetime_mapping[name] = "Date-Time"
            self.data["Date-Time"] = pd.to_datetime(self.data["Date-Time"], errors='coerce')
            self.unique_datetime = True
        elif len(self.datetime_mapping) > 1:
            print(f"Multiple Date-Time columns found: {', '.join(self.datetime_mapping.values())}. Every signal will have its own Date-Time column.")
            self.unique_datetime = False
        elif len(self.datetime_mapping) == 0:
            print("No Date-Time columns found. Please check the dataset.")
            self.unique_datetime = False
        if "Date-Time" not in self.data.columns:
            print("No Date-Time column found in the data. Please check the dataset.")
        return self

    def show(self):
        """Plot the data in the dataset."""
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        fig.suptitle(f"{self.name} Dataset", fontsize=16, fontweight='bold')
        if not hasattr(self, "data") or self.data.empty:
            print("No data to show. Please load the dataset first.")
            return
        if not hasattr(self, "info") or self.info.empty:
            print("No info to show. Please load the dataset first.")
            return
        axes = axes.flatten()
        if hasattr(self, "info"):
            cnt = 0
            for index, signal in enumerate(self.data.columns):
                if "Date-Time" in signal:
                    continue
                if cnt > 5:
                    print("More signals than subplots available. Please increase the number of subplots.")
                    break
                axes[cnt].plot(self.data[self.datetime_mapping[signal]], self.data[signal], "o")
                axes[cnt].tick_params(axis='x', rotation=45)
                if "PRM" in signal:
                    axes[cnt].set_title(fr"$e^-$-lifetime")
                    axes[cnt].set_ylabel(fr"$e^-$-lifetime [s]")
                elif "AE" in signal:
                    axes[cnt].set_title(fr"$H_2$O Concentration")
                    axes[cnt].set_ylabel(fr"$H_2$O Concentration [ppb]")
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

    def findTimes(self, scan_time=10, threshold=0.995, manual=False):
        """
        Finds the maximum and minimum liquid levels and their corresponding times, as well as the start time of a significant drop in liquid level within a specified time window.

        Parameters
        ----------
        scan_time : int or float, optional
            The time window (in minutes) after the maximum liquid level to search for the minimum liquid level. Default is 60 minutes. It is used to define the region where the purification is still one before it is turned off.
        threshold : float, optional
            The threshold (as a fraction of the minimum level) to determine the start of a significant drop in liquid level. Default is 0.995.

        Returns
        -------
        self : object
            The instance with the following attributes set:
                - max_level : float
                    The maximum liquid level found in the data.
                - max_time : pandas.Timestamp
                    The timestamp at which the maximum liquid level occurs.
                - min_level : float
                    The minimum liquid level found within the specified time window after the maximum.
                - min_time : pandas.Timestamp
                    The timestamp at which the minimum liquid level occurs within the specified time window.
                - start_time : pandas.Timestamp
                    The timestamp at which the liquid level first drops below the specified threshold of the minimum level.
                - end_time : pandas.Timestamp
                    The timestamp one day after the start_time, marking the end of the drop period.

        Raises
        ------
        ValueError
            If required data is missing, empty, or if the specified time window is invalid.
        TypeError
            If input types are incorrect.
        """
        finding_time = datetime.timedelta(minutes=scan_time)
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if self.data.empty:
            raise ValueError("The data is empty. Please load the dataset first.")
        if "PAB_S1_LT_13_AR_REAL_F_CV" not in self.data.columns:
            raise ValueError("The required column 'PAB_S1_LT_13_AR_REAL_F_CV' is not present in the data.")
        if not isinstance(finding_time, datetime.timedelta):
            raise TypeError("finding_time must be a datetime.timedelta object.")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be an int or float.")

        # Finds the max liquid level and the time it occurs
        self.max_level = self.data["PAB_S1_LT_13_AR_REAL_F_CV"].max()
        self.max_time = self.data.loc[self.data["PAB_S1_LT_13_AR_REAL_F_CV"] == self.max_level][self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]].iloc[0]

        # Finds the min liquid level and the time it occurs after the max time
        if self.max_time + finding_time > self.data[self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]].iloc[-1]:
            raise ValueError("The max time plus finding time exceeds the data range.")
        if self.data[self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]].iloc[-1] - self.max_time < finding_time:
            raise ValueError("The finding time exceeds the data range after the max time.")
        min_mask = (self.data[self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]] > self.max_time) & (self.data[self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]] < self.max_time + finding_time)
        if self.data["PAB_S1_LT_13_AR_REAL_F_CV"].loc[min_mask].empty:
            raise ValueError("No data available in the specified time range after the max time.")
        self.min_level = self.data["PAB_S1_LT_13_AR_REAL_F_CV"].loc[min_mask].min()
        self.min_time = self.data.loc[
            (self.data[self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]] > self.max_time) & (self.data["PAB_S1_LT_13_AR_REAL_F_CV"] == self.min_level)
        ].index[0]

        # Finds the start of the drop in liquid level
        if self.min_level == float('inf'):
            raise ValueError("No minimum liquid level found in the specified time range after the max time.")
        drop_mask = self.data["PAB_S1_LT_13_AR_REAL_F_CV"] < threshold * self.min_level
        self.start_time = self.data.loc[drop_mask][self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]].iloc[0]
        if manual == True:
            self.start_time = input(f"Please enter the start time of the filter stop (default is {self.start_time}, use same format): ")
            self.start_time = pd.to_datetime(self.start_time)
        self.end_time = self.start_time + datetime.timedelta(days=1)
        if self.end_time > max(self.data[self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]]):
            # print("End time exceeds the data range. Setting end time to the last available time in the data.")
            self.end_time = max(self.data[self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]])
        if self.start_time > self.end_time:
            raise ValueError("Start time cannot be after end time.")

        return self

    def level(self, ax=None, color=None, integration_time=60, fit_legend=False):
        """
        Plots the liquid level over time and highlights the maximum and minimum levels.

        Parameters
        ----------
        show : bool, optional
            If True, displays the plot (default is False).
        ax : matplotlib.axes.Axes, optional
            If provided, uses this axes for plotting instead of creating a new one.
        color : str, optional
            Color for the plot line (default is None, which uses the default color).

        Returns
        -------
        self : object
            Returns self with the plotted liquid level data.
        """
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        if "PAB_S1_LT_13_AR_REAL_F_CV" not in self.data.columns:
            raise ValueError("The required column 'PAB_S1_LT_13_AR_REAL_F_CV' is not present in the data.")

        dt_col = self.datetime_mapping["PAB_S1_LT_13_AR_REAL_F_CV"]
        self.level_data = self.data.set_index(dt_col)["PAB_S1_LT_13_AR_REAL_F_CV"]
        if ax is None:
            fig, ax = plt.subplots()

        self.level_ini = self.level_data.loc[
            (self.level_data.index > self.start_time) &
            (self.level_data.index < self.start_time + datetime.timedelta(minutes=integration_time))
        ].mean()
        self.level_ini_err = self.level_data.loc[
            (self.level_data.index > self.start_time) &
            (self.level_data.index < self.start_time + datetime.timedelta(minutes=integration_time))
        ].std()
        self.level_end = self.level_data.loc[
            (self.level_data.index > self.end_time - datetime.timedelta(minutes=integration_time)) &
            (self.level_data.index < self.end_time)
        ].mean()
        self.level_end_err = self.level_data.loc[
            (self.level_data.index > self.end_time - datetime.timedelta(minutes=integration_time)) &
            (self.level_data.index < self.end_time)
        ].std()
        if fit_legend:
            ax.plot((self.level_data.index - self.start_time).total_seconds(), self.level_data, label=f'{self.name}-> Initial Level: {self.level_ini:.2f} ± {self.level_ini_err:.2f} in; Final Level: {self.level_end:.2f} ± {self.level_end_err:.2f} in', color=f'tab:{color}')
            ax.axvline(x=0, color=f'tab:{color}', linestyle='--', label=f'Start: {self.start_time}')
        else:
            ax.plot((self.level_data.index - self.start_time).total_seconds(), self.level_data, label=f'{self.name}', color=f'tab:{color}')
        # ax.axvline(x=(self.end_time-self.start_time).total_seconds(), color=f'tab:{color}', linestyle='--', label=f'End Time: {self.end_time}')

        ax.set_xlabel("Time since start [s]")
        ax.set_ylabel("Liquid Level [in]")
        ax.set_title("Liquid Level Over Time")
        ax.tick_params(axis='x', rotation=0)
        ax.grid()
        ax.legend()

        return self

    def h2oConcentration(self, integration_time_ini=60, integration_time_end=3*60, offset_ini=60, offset_end=8*60, show=False, ax=None, color=None, signal_name="PAB_S1_AE_611_AR_REAL_F_CV"):
        """
        Calculates the H2O concentration in parts per billion (ppb) based on the AE signal data within specified integration and offset time windows.

        Parameters
        ----------
        integration_time_ini : int, optional
            Integration time in minutes for the initial window (default is 60).
        integration_time_end : int, optional
            Integration time in minutes for the final window (default is 180).
        offset_ini : int, optional
            Offset in minutes from the start time for the initial window (default is 60).
        offset_end : int, optional
            Offset in minutes from the end time for the final window (default is 60).
        show : bool, optional
            If True, displays a plot of the H2O concentration over time with highlighted integration windows (default is False).

        Returns
        -------
        self : object
            Returns self with calculated H2O concentration statistics as attributes.

        Raises
        ------
        ValueError
            If the required data is not loaded or the necessary column is missing from the dataset.

        Notes
        -----
        - The method computes the mean and standard deviation of H2O concentration in two time windows: one at the beginning and one at the end of the selected period.
        - If `show` is True, a plot is generated to visualize the H2O concentration and the integration windows.
        """
        self.integration_time_ini = datetime.timedelta(minutes=integration_time_ini)
        self.integration_time_end = datetime.timedelta(minutes=integration_time_end)
        self.offset_ini = datetime.timedelta(minutes=offset_ini)
        self.offset_end = datetime.timedelta(minutes=offset_end)
        if self.integration_time_ini > self.offset_ini:
            print("Warning: integration_time_ini is greater than offset_ini. This may lead to unexpected results.")
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        if "PAB_S1_AE_611_AR_REAL_F_CV" not in self.data.columns:
            raise ValueError(f"The required column {signal_name} is not present in the data.")
        dt_col = self.datetime_mapping[signal_name]
        mask = (self.data[dt_col] > self.start_time - self.offset_ini) & (self.data[dt_col] < self.end_time)
        self.h20_data = self.data.loc[mask, [dt_col, signal_name]].set_index(dt_col)[signal_name]
        # self.h20_data = self.data.loc[(self.data.index > self.start_time) & (self.data.index < self.end_time)][signal_name]
        self.h20_ini = self.h20_data.loc[
            (self.h20_data.index > self.start_time - self.integration_time_ini) &
            (self.h20_data.index < self.start_time)
        ].mean()
        self.h20_ini_err = self.h20_data.loc[
            (self.h20_data.index > self.start_time - self.integration_time_ini) &
            (self.h20_data.index < self.start_time)
        ].std()
        self.h20_end = self.h20_data.loc[
            (self.h20_data.index > self.end_time - self.offset_end - self.integration_time_end) &
            (self.h20_data.index < self.end_time - self.offset_end)
        ].mean()
        self.h20_end_err = self.h20_data.loc[
            (self.h20_data.index > self.end_time - self.offset_end - self.integration_time_end) &
            (self.h20_data.index < self.end_time - self.offset_end)
        ].std()

        if show:
            if ax is None:
                fig, ax = plt.subplots()
            elif ax is not None:
                ax = ax
            if not hasattr(self, 'h20_data') or self.h20_data.empty:
                print("No H2O data to show. Please load the dataset first.")
                return self
            ax.plot((self.h20_data.index - self.start_time).total_seconds(), self.h20_data, label=fr'{self.name}', color=f'tab:{color}')
            # ax.axvline(x=self.offset_ini.total_seconds(), color=f'tab:{color}', linestyle='--')
            # ax.axvline(x=(self.offset_ini + self.integration_time_ini).total_seconds(), color=f'tab:{color}', linestyle='--')
            # ax.axvline(x=(self.end_time - self.offset_end - self.integration_time_end - self.start_time).total_seconds(), color=f'tab:{color}', linestyle='--')
            # ax.axvline(x=(self.end_time - self.offset_end - self.start_time).total_seconds(), color=f'tab:{color}', linestyle='--')
            ax.axhline(y=self.h20_ini, color=f'dark{color}', linestyle='--', label=fr'$\mu$={self.h20_ini:.1f} $\pm$ {self.h20_ini_err:.1f} ppb')
            ax.axhline(y=self.h20_end, color=f'tab:{color}', linestyle='--', label=fr'$\mu$={self.h20_end:.1f} $\pm$ {self.h20_end_err:.1f} ppb')
            ax.set_xlabel('Time since start [s]')
            ax.set_ylabel('H2O Concentration [ppb]')
            ax.set_title(fr'$H^{2}O$ Concentration Over Time')
            ax.legend()
            ax.tick_params(axis='x', rotation=0)
            if ax is None:
                fig.suptitle(f"{self.name} H2O Concentration", fontsize=16, fontweight='bold')
                fig.tight_layout()
                plt.show()
        return self

    def temperature(self, integration_time_ini=60, integration_time_end=3*60, offset_ini=60, offset_end=8*60, show=False, ax=None, color=None):
        """
        Calculates the temperature statistics based on the TE signal data within specified integration and offset time windows.

        Parameters
        ----------
        integration_time_ini : int, optional
            Integration time in minutes for the initial window (default is 60).
        integration_time_end : int, optional
            Integration time in minutes for the final window (default is 180).
        offset_ini : int, optional
            Offset in minutes from the start time for the initial window (default is 60).
        offset_end : int, optional
            Offset in minutes from the end time for the final window (default is 60).
        show : bool, optional
            If True, displays a plot of the temperature over time with highlighted integration windows (default is False).

        Returns
        -------
        self : object
            Returns self with calculated temperature statistics as attributes.

        Raises
        ------
        ValueError
            If the required data is not loaded or the necessary column is missing from the dataset.

        Notes
        -----
        - The method computes the mean and standard deviation of temperature in two time windows: one at the beginning and one at the end of the selected period.
        - If `show` is True, a plot is generated to visualize the temperature and the integration windows.
        """
        self.integration_time_ini = datetime.timedelta(minutes=integration_time_ini)
        self.integration_time_end = datetime.timedelta(minutes=integration_time_end)
        self.offset_ini = datetime.timedelta(minutes=offset_ini)
        self.offset_end = datetime.timedelta(minutes=offset_end)
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Data not loaded. Please load the dataset first.")
        if "PAB_S1_TE_324_AR_REAL_F_CV" not in self.data.columns:
            raise ValueError(f"{self.name}: The required column 'PAB_S1_TE_324_AR_REAL_F_CV' is not present in the data. Available columns are: " + ", ".join(self.data.columns))
        dt_col = self.datetime_mapping["PAB_S1_TE_324_AR_REAL_F_CV"]
        mask = (self.data[dt_col] > self.start_time - self.offset_ini) & (self.data[dt_col] < self.end_time)
        self.temp_data = self.data.loc[mask, [dt_col, "PAB_S1_TE_324_AR_REAL_F_CV"]].set_index(dt_col)["PAB_S1_TE_324_AR_REAL_F_CV"]
        # self.temp_data = self.data.loc[(self.data.index > self.start_time) & (self.data.index < self.end_time)]["PAB_S1_TE_324_AR_REAL_F_CV"]
        self.temp_ini = self.temp_data.loc[
            (self.temp_data.index > self.start_time - self.integration_time_ini) &
            (self.temp_data.index < self.start_time)
        ].mean()
        self.temp_ini_err = self.temp_data.loc[
            (self.temp_data.index > self.start_time - self.integration_time_ini) &
            (self.temp_data.index < self.start_time)
        ].std()
        self.temp_end = self.temp_data.loc[
            (self.temp_data.index > self.end_time - self.offset_end - self.integration_time_end) &
            (self.temp_data.index < self.end_time - self.offset_end)
        ].mean()
        self.temp_end_err = self.temp_data.loc[
            (self.temp_data.index > self.end_time - self.offset_end - self.integration_time_end) &
            (self.temp_data.index < self.end_time - self.offset_end)
        ].std()

        if show:
            if ax is None:
                fig, ax = plt.subplots()
            elif ax is not None:
                ax = ax
            if not hasattr(self, 'temp_data') or self.temp_data.empty:
                print("No Temperature data to show. Please load the dataset first.")
                return self
            ax.plot((self.temp_data.index - self.start_time).total_seconds(), self.temp_data, label=f'{self.name} Temperature', color=f"tab:{color}")
            ax.axhline(y=self.temp_ini, color=f'{color}', linestyle='--', label=fr'$\mu$={self.temp_ini:.1f} $\pm$ {self.temp_ini_err:.1f} K')
            ax.axhline(y=self.temp_end, color=f'tab:{color}', linestyle='--', label=fr'$\mu$={self.temp_end:.1f} $\pm$ {self.temp_end_err:.1f} K')
            # ax.axvline(x=self.offset_ini.total_seconds(), color=f'{color}', linestyle='--')
            # ax.axvline(x=(self.offset_ini + self.integration_time_ini).timestamp(), color=f'{color}', linestyle='--')
            # ax.axvline(x=(self.end_time - self.offset_end - self.integration_time_end - self.start_time).timestamp(), color=f'{color}', linestyle='--')
            # ax.axvline(x=(self.end_time - self.offset_end - self.start_time).timestamp(), color=f'{color}', linestyle='--')
            ax.set_xlabel('Time since start [s]')
            ax.set_ylabel('Temperature [K]')
            ax.set_title('Sample temperature Over Time')
            ax.legend()
            ax.tick_params(axis='x', rotation=0)
            if ax is None:
                fig.suptitle(f"{self.name} Temperature", fontsize=16, fontweight='bold')
                fig.tight_layout()
                plt.show()
        return self

    def purity(self, offset_ini=60, show=False, ax=None, color=None, fit_legend=False):
        """
        Calculates the purity of the data based on the Luke PRM lifetime signal within the specified time window. The used function is an exponential decay function fitted to the purity data.
        Parameters
        ----------
        show : bool, optional
            If True, displays a plot of the purity over time with the fitted exponential curve (default is False).
        Returns
        -------
        self : object
            Returns self with calculated purity statistics as attributes. All units in seconds.
        Raises
        -------
        ValueError
            If the purity data is empty or not found.
        """
        self.offset_ini = datetime.timedelta(minutes=offset_ini)
        dt_col = self.datetime_mapping["Luke_PRM_LIFETIME_F_CV"]
        mask = (self.data[dt_col] > self.start_time - self.offset_ini) & (self.data[dt_col] < self.end_time)
        self.purity = self.data.loc[mask, [dt_col, "Luke_PRM_LIFETIME_F_CV"]].set_index(dt_col)["Luke_PRM_LIFETIME_F_CV"]
        if self.purity.empty:
            raise ValueError("Purity data is empty. Please ensure the dataset is loaded and contains the 'Luke_PRM_LIFETIME_F_CV' column.")
        self.purity = self.purity.dropna()
        if self.purity.empty:
            raise ValueError("Purity data is empty. Please ensure the dataset is loaded and contains the 'Luke_PRM_LIFETIME_F_CV' column.")
        self.popt, self.pcov = curve_fit(self.exp, (self.purity.index-self.purity.index[0]).total_seconds(), self.purity.values,
                                         p0=[self.purity.iloc[0], 8*3600, self.purity.iloc[-1]])
        if show == True:
            if not hasattr(self, 'purity') or self.purity.empty:
                print("No Purity data to show. Please load the dataset first.")
                return self
            ax.plot((self.purity.index - self.start_time).total_seconds(), self.purity, "-o", label=fr"{self.name}", markersize=5.0, color=f"tab:{color}")
            if fit_legend:
                ax.plot((self.purity.index - self.start_time).total_seconds(), self.exp((self.purity.index-self.purity.index[0]).total_seconds(), *self.popt), label=fr'Fit: A={1e3*self.popt[0]:.2f} $\pm$ {1e3*np.sqrt(np.diag(self.pcov))[0]:.2f} ms, $\tau$={(1/3600)*self.popt[1]:.2f} $\pm$ {(1/3600)*np.sqrt(np.diag(self.pcov))[1]:.2f} h, C={1e3*self.popt[2]:.2f} $\pm$ {1e3*np.sqrt(np.diag(self.pcov))[2]:.2f} ms', linestyle='--', color=f"tab:{color}")
                ax.set_title(r'$e^-$ lifetime Over Time - Fit: $Ae^{\frac{x}{\tau}} + C$')
            else:
                # ax.plot((self.purity.index - self.start_time).total_seconds(), self.exp((self.purity.index-self.purity.index[0]).total_seconds(), *self.popt), linestyle='--', color=f"tab:{color}")
                ax.set_title(r'$e^-$ lifetime Over Time')
            ax.set_xlabel('Time since start [s]')
            ax.set_ylabel(r'$e^-$ lifetime [s]')

            ax.legend(ncol=1)
            ax.tick_params(axis='x', rotation=0)
        return self