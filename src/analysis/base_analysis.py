"""
Abstract base class for analysis operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict


class BaseAnalysis(ABC):
    """
    Abstract base class for analysis operations.

    This class defines the interface that all analysis classes must implement.
    Each analysis class handles a specific type of signal analysis.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the analysis class.

        Args:
            name: Optional name for this analysis instance
        """
        self.name = name
        self._results: Dict[str, Any] = {}

    @abstractmethod
    def analyze(self, **kwargs) -> Any:
        """
        Perform the analysis operation.

        Args:
            **kwargs: Additional parameters for the analysis

        Returns:
            Analysis results (format depends on the specific analysis)
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """
        Get the analysis results.

        Returns:
            Copy of the current results dictionary
        """
        return self._results.copy()
