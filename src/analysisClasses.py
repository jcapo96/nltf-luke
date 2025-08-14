from dataClasses import Dataset
import numpy as np

class Analysis:
    """
    The Analysis class provides methods to analyze and visualize data from three different datasets: baseline, ullage, and liquid.
    Each dataset is expected to be represented by a Dataset object, which is initialized with a specific file path.
    Attributes:
        name (str): Optional name identifier for the analysis.
        baseline (Dataset): Dataset object for the baseline data.
        ullage (Dataset): Dataset object for the ullage data.
        liquid (Dataset): Dataset object for the liquid data.
    Methods:
        __init__(path: str, name=None):
            Initializes the Analysis object with the given path and optional name.
            Loads the baseline, ullage, and liquid datasets from Excel files located at the specified path.
        purity(show=False, ax=None):
            Loads and processes the datasets, then plots the purity for baseline, ullage, and liquid.
            Args:
                show (bool): Whether to display the plot immediately. Default is False.
                ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure and axis are created.
        temperature(show=False, ax=None):
            Loads and processes the datasets, then plots the temperature for baseline, ullage, and liquid.
            Args:
                show (bool): Whether to display the plot immediately. Default is False.
                ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure and axis are created.
        h2oConcentration(show=False, ax=None):
            Loads and processes the datasets, then plots the H2O concentration for baseline, ullage, and liquid.
            Args:
                show (bool): Whether to display the plot immediately. Default is False.
                ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure and axis are created.
    """

    def __init__(self, path: str, name=None):
        self.name = name
        self.baseline = Dataset(path=fr"{path}/{self.name}_baseline.xlsx")
        self.ullage = Dataset(path=fr"{path}/{self.name}_ullage.xlsx")
        self.liquid = Dataset(path=fr"{path}/{self.name}_liquid.xlsx")

    def purity(self, show=False, ax=None, fit_legend=False, manual=False):
        """
        Plots the purity for baseline, ullage, and liquid datasets.
        This method loads the data for each region (baseline, ullage, liquid),
        computes their respective times, and then plots their purity on the same
        matplotlib axis. Each region is plotted with a distinct color.
        Parameters
        ----------
        show : bool, optional
            If True, the plot will be displayed. Default is False.
        ax : matplotlib.axes.Axes, optional
            The matplotlib axis to plot on. If None, a new figure and axis will be created.
        Returns
        -------
        None
        Notes
        -----
        - The method assumes that `self.baseline`, `self.ullage`, and `self.liquid`
          are objects with `load()`, `findTimes()`, and `purity()` methods.
        - The `purity()` method of each region is expected to accept `show`, `ax`, and `color` arguments.
        """

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
        self.baseline.load()
        self.ullage.load()
        self.liquid.load()

        self.baseline.assign_datetime()
        self.ullage.assign_datetime()
        self.liquid.assign_datetime()

        self.baseline.findTimes(manual=manual)
        self.ullage.findTimes(manual=manual)
        self.liquid.findTimes(manual=manual)

        print("Plotting purity for baseline, ullage, and liquid datasets...")
        print(f"Baseline start time: {self.baseline.start_time}, end time: {self.baseline.end_time}")
        print(f"Ullage start time: {self.ullage.start_time}, end time: {self.ullage.end_time}")
        print(f"Liquid start time: {self.liquid.start_time}, end time: {self.liquid.end_time}")

        colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }
        self.baseline.purity(show=show, ax=ax, color=colors['baseline'], fit_legend=fit_legend)
        self.ullage.purity(show=show, ax=ax, color=colors['ullage'], fit_legend=fit_legend)
        self.liquid.purity(show=show, ax=ax, color=colors['liquid'], fit_legend=fit_legend)

        print(f"Baseline -> Initial Purity: {1e3*self.baseline.exp(0, *self.baseline.popt):.0f} ms; Final Purity: {1e3*self.baseline.popt[2]:.0f} ms; ΔPurity: {1e3*(self.baseline.popt[2] - self.baseline.exp(0, *self.baseline.popt)):.0f} ms; τ: {(1/3600)*self.baseline.popt[1]:.0f} h")
        print(f"Ullage -> Initial Purity: {1e3*self.ullage.exp(0, *self.ullage.popt):.0f} ms; Final Purity: {1e3*self.ullage.popt[2]:.0f} ms; ΔPurity: {1e3*(self.ullage.popt[2] - self.ullage.exp(0, *self.ullage.popt)):.0f} ms; τ: {(1/3600)*self.ullage.popt[1]:.0f} h")
        print(f"Liquid -> Initial Purity: {1e3*self.liquid.exp(0, *self.liquid.popt):.0f} ms; Final Purity: {1e3*self.liquid.popt[2]:.0f} ms; ΔPurity: {1e3*(self.liquid.popt[2] - self.liquid.exp(0, *self.liquid.popt)):.0f} ms; τ: {(1/3600)*self.liquid.popt[1]:.0f} h")
        print("Purity plotted successfully.")

        return self

    def temperature(self, show=False, ax=None, manual=False):
        """
        Plots the temperature profiles for baseline, ullage, and liquid components.
        This method loads and processes temperature data for the baseline, ullage, and liquid
        components, and plots their temperature profiles on a single matplotlib axis. Each component
        is plotted with a distinct color. If the 'liquid' component is unavailable or fails to plot,
        only baseline and ullage are shown.
        Args:
            show (bool, optional): If True, displays the plot. Defaults to False.
            ax (matplotlib.axes.Axes, optional): An existing matplotlib axis to plot on. If None,
                a new figure and axis are created.
        Notes:
            - The method assumes that the baseline, ullage, and liquid objects have 'load', 'findTimes',
              and 'temperature' methods.
            - The legend is updated to reflect the number of successfully plotted components.
        """

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }

        plotted = []

        # Baseline
        try:
            self.baseline.load()
            self.baseline.assign_datetime()
            self.baseline.findTimes(manual=manual)
            self.baseline.temperature(show=show, ax=ax, color=colors['baseline'])
            plotted.append('baseline')
        except Exception as e:
            print(f"Baseline temperature plot failed: {e}")

        # Ullage
        try:
            self.ullage.load()
            self.ullage.assign_datetime()
            self.ullage.findTimes(manual=manual)
            self.ullage.temperature(show=show, ax=ax, color=colors['ullage'])
            plotted.append('ullage')
        except Exception as e:
            print(f"Ullage temperature plot failed: {e}")

        # Liquid
        try:
            self.liquid.load()
            self.liquid.assign_datetime()
            self.liquid.findTimes(manual=manual)
            self.liquid.temperature(show=show, ax=ax, color=colors['liquid'])
            plotted.append('liquid')
        except Exception as e:
            print(f"Liquid temperature plot failed: {e}")

        if plotted:
            ax.legend(ncol=len(plotted))
        return self

    def h2oConcentration(self, show=False, ax=None, manual=False, integration_time_ini=60, integration_time_end=60, offset_ini=1, offset_end=8):
        """
        Plots the H2O concentration for baseline, ullage, and liquid regions.
        This method loads the data for baseline, ullage, and liquid, finds their respective times,
        and then plots the H2O concentration for each region on the same matplotlib axis. Each region
        is plotted with a distinct color: blue for baseline, red for ullage, and green for liquid.
        If no axis is provided, a new matplotlib figure and axis are created.
        Parameters
        ----------
        show : bool, optional
            Whether to display the plot immediately after plotting. Default is False.
        ax : matplotlib.axes.Axes, optional
            The matplotlib axis to plot on. If None, a new figure and axis are created.
        Returns
        -------
        None
        Notes
        -----
        This method assumes that the `baseline`, `ullage`, and `liquid` attributes have
        `load`, `findTimes`, and `h2oConcentration` methods implemented.
        """

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
        self.baseline.load()
        self.ullage.load()
        self.liquid.load()

        self.baseline.assign_datetime()
        self.ullage.assign_datetime()
        self.liquid.assign_datetime()

        self.baseline.findTimes(manual=manual)
        self.ullage.findTimes(manual=manual)
        self.liquid.findTimes(manual=manual)

        colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }
        self.baseline.h2oConcentration(show=show, ax=ax, color=colors['baseline'], integration_time_ini=integration_time_ini, integration_time_end=integration_time_end, offset_ini=offset_ini, offset_end=offset_end)
        self.ullage.h2oConcentration(show=show, ax=ax, color=colors['ullage'], integration_time_ini=integration_time_ini, integration_time_end=integration_time_end, offset_ini=offset_ini, offset_end=offset_end)
        self.liquid.h2oConcentration(show=show, ax=ax, color=colors['liquid'], integration_time_ini=integration_time_ini, integration_time_end=integration_time_end, offset_ini=offset_ini, offset_end=offset_end)
        ax.set_ylim(0, None)

        print("Plotting H2O concentration for baseline, ullage, and liquid datasets...")
        print(fr"Baseline -> Initial H2O: {self.baseline.h20_ini:.2f} ± {self.baseline.h20_ini_err:.2f} ppb; Final H2O: {self.baseline.h20_end:.2f} ± {self.baseline.h20_end_err:.2f} ppb; ΔH2O: {(self.baseline.h20_end - self.baseline.h20_ini):.2f} ± {np.sqrt(self.baseline.h20_end_err**2 + self.baseline.h20_ini_err**2):.2f} ppb")
        print(fr"Ullage -> Initial H2O: {self.ullage.h20_ini:.2f} ± {self.ullage.h20_ini_err:.2f} ppb; Final H2O: {self.ullage.h20_end:.2f} ± {self.ullage.h20_end_err:.2f} ppb; ΔH2O: {(self.ullage.h20_end - self.ullage.h20_ini):.2f} ± {np.sqrt(self.ullage.h20_end_err**2 + self.ullage.h20_ini_err**2):.2f} ppb")
        print(fr"Liquid -> Initial H2O: {self.liquid.h20_ini:.2f} ± {self.liquid.h20_ini_err:.2f} ppb; Final H2O: {self.liquid.h20_end:.2f} ± {self.liquid.h20_end_err:.2f} ppb; ΔH2O: {(self.liquid.h20_end - self.liquid.h20_ini):.2f} ± {np.sqrt(self.liquid.h20_end_err**2 + self.liquid.h20_ini_err**2):.2f} ppb")
        print("H2O concentration plotted successfully.")
        ax.legend(ncol=3)
        return self

    def level(self, ax=None, manual=False, fit_legend=False):
        """
        Plots the level for baseline, ullage, and liquid datasets.
        This method loads the data for each region (baseline, ullage, liquid),
        computes their respective times, and then plots their levels on the same
        matplotlib axis. Each region is plotted with a distinct color.
        Parameters
        ----------
        show : bool, optional
            If True, the plot will be displayed. Default is False.
        ax : matplotlib.axes.Axes, optional
            The matplotlib axis to plot on. If None, a new figure and axis will be created.
        Returns
        -------
        None
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
        self.baseline.load()
        self.ullage.load()
        self.liquid.load()

        self.baseline.assign_datetime()
        self.ullage.assign_datetime()
        self.liquid.assign_datetime()

        self.baseline.findTimes(manual=manual)
        self.ullage.findTimes(manual=manual)
        self.liquid.findTimes(manual=manual)

        colors = {
            'baseline': 'blue',
            'ullage': 'red',
            'liquid': 'green'
        }
        self.baseline.level(ax=ax, color=colors['baseline'], fit_legend=fit_legend)
        self.ullage.level(ax=ax, color=colors['ullage'], fit_legend=fit_legend)
        self.liquid.level(ax=ax, color=colors['liquid'], fit_legend=fit_legend)

        ax.legend(ncol=1)
        return self