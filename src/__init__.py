"""
NLTF-LUKE Data Analysis Framework

This package provides a modular framework for analyzing NLTF data with improved structure
and separation of concerns.

Main Classes:
- Dataset: Main dataset class for handling Excel data files
- Analysis: Main analysis class for coordinating multi-dataset analysis
- DataValidator: Utility class for data validation
- BaseDataProcessor: Abstract base class for data processors
- LiquidLevelProcessor: Processes liquid level data
- H2OConcentrationProcessor: Processes H2O concentration data
- TemperatureProcessor: Processes temperature data
- PurityProcessor: Processes electron lifetime (purity) data
- DatasetManager: Manages multiple datasets for analysis
- BaseAnalysis: Abstract base class for analysis operations
- PurityAnalysis: Analyzes purity across datasets
- TemperatureAnalysis: Analyzes temperature across datasets
- H2OConcentrationAnalysis: Analyzes H2O concentration across datasets
- LiquidLevelAnalysis: Analyzes liquid level across datasets
"""

from .dataClasses import (
    Dataset,
    DataValidator,
    BaseDataProcessor,
    LiquidLevelProcessor,
    H2OConcentrationProcessor,
    TemperatureProcessor,
    PurityProcessor
)

from .analysisClasses import (
    Analysis,
    BaseAnalysis,
    DatasetManager,
    PurityAnalysis,
    TemperatureAnalysis,
    H2OConcentrationAnalysis,
    LiquidLevelAnalysis
)

__version__ = "2.0.0"
__author__ = "NLTF-LUKE Team"

__all__ = [
    # Main classes
    'Dataset',
    'Analysis',

    # Data processing classes
    'DataValidator',
    'BaseDataProcessor',
    'LiquidLevelProcessor',
    'H2OConcentrationProcessor',
    'TemperatureProcessor',
    'PurityProcessor',

    # Analysis classes
    'BaseAnalysis',
    'DatasetManager',
    'PurityAnalysis',
    'TemperatureAnalysis',
    'H2OConcentrationAnalysis',
    'LiquidLevelAnalysis',
]
