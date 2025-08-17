"""
Base classes for data processing and analysis.
"""

from abc import ABC, abstractmethod
from typing import Any
from .standard_format import StandardDataFormat


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processing operations.

    This class defines the interface that all signal processors must implement.
    Each processor handles a specific type of signal (liquid level, H2O concentration, etc.).
    """

    def __init__(self, standard_data: StandardDataFormat):
        """
        Initialize the processor with standard format data.

        Args:
            standard_data: StandardDataFormat object containing the data to process
        """
        self.standard_data = standard_data
        self._validate_data()

    @abstractmethod
    def process(self, **kwargs) -> Any:
        """
        Process the data and return results.

        Args:
            **kwargs: Additional parameters for processing

        Returns:
            Processed data results (format depends on the specific processor)
        """
        pass

    @abstractmethod
    def _validate_data(self):
        """
        Validate that required data is available.

        This method should check if the required signal data exists
        and set appropriate flags (e.g., self.has_signal = True/False).
        """
        pass
