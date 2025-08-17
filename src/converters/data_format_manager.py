"""
Manages data format conversion and provides a unified interface.
"""

from typing import List
from .base_converter import BaseDataConverter
from .seeq_new_converter import SeeqNewConverter
from .seeq_old_converter import SeeqOldConverter
from core.standard_format import StandardDataFormat


class DataFormatManager:
    """
    Manages data format conversion and provides a unified interface.

    This class acts as a central registry for all available converters.
    It automatically selects the appropriate converter for each file type.

    To add a new converter:
    1. Create a class that inherits from BaseDataConverter
    2. Implement the required methods
    3. Register it with this manager using register_converter()
    """

    def __init__(self):
        """Initialize the manager and register default converters."""
        self.converters: List[BaseDataConverter] = []
        self._register_default_converters()

    def _register_default_converters(self):
        """Register the default converters that come with the framework."""
        self.register_converter(SeeqNewConverter())
        self.register_converter(SeeqOldConverter())

    def register_converter(self, converter: BaseDataConverter):
        """
        Register a new data converter.

        Args:
            converter: Instance of a class that inherits from BaseDataConverter

        This method allows you to add custom converters to the framework.
        Converters are tried in the order they are registered.
        """
        self.converters.append(converter)

    def convert_file(self, file_path: str) -> StandardDataFormat:
        """
        Convert a file to standard format using the appropriate converter.

        Args:
            file_path: Path to the file to convert

        Returns:
            StandardDataFormat object with converted data

        This method automatically selects the first converter that can handle the file.
        If no converter is found, it raises a helpful error message.
        """
        for converter in self.converters:
            if converter.can_convert(file_path):
                return converter.convert(file_path)

        # If no converter found, provide helpful error message
        supported_formats = [f"{converter.__class__.__name__}" for converter in self.converters]
        raise ValueError(
            f"No converter found for file {file_path}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    def can_convert(self, file_path: str) -> bool:
        """
        Check if any converter can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if any converter can handle the file, False otherwise
        """
        for converter in self.converters:
            if converter.can_convert(file_path):
                return True
        return False

    def get_converter(self, file_path: str) -> BaseDataConverter:
        """
        Get the converter that can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            The first converter that can handle the file

        Raises:
            ValueError: If no converter can handle the file
        """
        for converter in self.converters:
            if converter.can_convert(file_path):
                return converter
        raise ValueError(f"No converter found for file {file_path}")

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported format names.

        Returns:
            List of converter class names that are currently registered
        """
        return [converter.__class__.__name__ for converter in self.converters]
