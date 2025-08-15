# NLTF-LUKE Framework Restructure

This document describes the restructured NLTF-LUKE data analysis framework, which has been reorganized for better maintainability, modularity, and separation of concerns.

## Overview of Changes

The original monolithic structure has been replaced with a modular, object-oriented design that separates data processing from analysis logic.

### Key Improvements

1. **Modular Design**: Each data type has its own specialized processor class
2. **Better Separation of Concerns**: Data loading, processing, and analysis are clearly separated
3. **Improved Error Handling**: More robust validation and error reporting
4. **Type Hints**: Added Python type hints for better code documentation
5. **Abstract Base Classes**: Clear interfaces for extending functionality
6. **Legacy Compatibility**: Original method names are preserved for backward compatibility

## New Class Structure

### Data Processing Classes (`dataClasses.py`)

#### Core Classes
- **`Dataset`**: Main dataset class that coordinates all data operations
- **`DataValidator`**: Utility class for validating file paths and data

#### Processor Classes
- **`BaseDataProcessor`**: Abstract base class for all data processors
- **`LiquidLevelProcessor`**: Handles liquid level data analysis and plotting
- **`H2OConcentrationProcessor`**: Processes H2O concentration measurements
- **`TemperatureProcessor`**: Analyzes temperature data
- **`PurityProcessor`**: Handles electron lifetime (purity) analysis with curve fitting

### Analysis Classes (`analysisClasses.py`)

#### Core Classes
- **`Analysis`**: Main analysis class that coordinates multi-dataset analysis
- **`DatasetManager`**: Manages loading and preparation of multiple datasets
- **`BaseAnalysis`**: Abstract base class for specialized analysis types

#### Specialized Analysis Classes
- **`PurityAnalysis`**: Analyzes purity across multiple datasets
- **`TemperatureAnalysis`**: Analyzes temperature profiles across datasets
- **`H2OConcentrationAnalysis`**: Analyzes H2O concentration across datasets
- **`LiquidLevelAnalysis`**: Analyzes liquid level changes across datasets

## Usage Examples

### Basic Dataset Usage

```python
from dataClasses import Dataset

# Create and load dataset
dataset = Dataset("path/to/data.xlsx")
dataset.load()
dataset.assign_datetime()

# Use specialized processors
if dataset.liquid_level is not None:
    # Find key time points
    max_level, max_time, min_level, min_time, start_time, end_time = \
        dataset.liquid_level.find_times(scan_time=10, threshold=0.995)
    
    # Plot liquid level
    dataset.liquid_level.plot_level(start_time, end_time, color='blue')
```

### Multi-Dataset Analysis

```python
from analysisClasses import Analysis

# Create analysis object
analysis = Analysis("path/to/data/directory", "EXPERIMENT_NAME")

# Run different types of analysis
analysis.purity(fit_legend=True)
analysis.temperature()
analysis.h2oConcentration()
analysis.level(fit_legend=True)

# Get all results
results = analysis.get_analysis_results()
```

### Custom Analysis

```python
from analysisClasses import DatasetManager

# Create dataset manager
dataset_manager = DatasetManager("path/to/data/directory", "EXPERIMENT_NAME")

# Prepare datasets
prepared_data = dataset_manager.prepare_datasets(manual=False)

# Custom analysis logic
for dataset_type, data in prepared_data.items():
    dataset = data['dataset']
    start_time, end_time = data['times'][4], data['times'][5]
    
    # Access specific processors
    if dataset.temperature is not None:
        temp_data = dataset.temperature.calculate_temperature(start_time, end_time)
        print(f"{dataset_type}: {temp_data['initial']:.1f} K → {temp_data['final']:.1f} K")
```

## Migration Guide

### For Existing Code

The restructured framework maintains backward compatibility through legacy method names:

- `dataset.findTimes()` → `dataset.liquid_level.find_times()`
- `dataset.level()` → `dataset.liquid_level.plot_level()`
- `dataset.h2oConcentration()` → `dataset.h2o_concentration.calculate_concentration()`
- `dataset.temperature()` → `dataset.temperature.calculate_temperature()`
- `dataset.purity()` → `dataset.purity.calculate_purity()`

### Recommended New Approach

Instead of using legacy methods, use the new processor-based approach:

```python
# Old way (still works)
dataset.findTimes()
dataset.level(ax=ax, color='blue')

# New way (recommended)
times = dataset.liquid_level.find_times()
dataset.liquid_level.plot_level(times[4], times[5], ax=ax, color='blue')
```

## Benefits of the New Structure

1. **Maintainability**: Each processor handles one specific data type
2. **Extensibility**: Easy to add new data types or analysis methods
3. **Testing**: Individual components can be tested in isolation
4. **Documentation**: Clear interfaces and responsibilities
5. **Error Handling**: Better validation and error reporting
6. **Performance**: More efficient data processing with specialized methods

## File Organization

```
src/
├── __init__.py              # Package initialization and exports
├── dataClasses.py           # Data processing classes
├── analysisClasses.py       # Analysis and coordination classes
├── example_usage.py         # Usage examples
└── README_RESTRUCTURE.md    # This documentation
```

## Future Enhancements

The new structure makes it easy to add:

- New data types (e.g., pressure, flow rate)
- Additional analysis methods
- Data export functionality
- Statistical analysis tools
- Custom plotting options
- Data validation rules

## Support

For questions or issues with the restructured framework, refer to the example usage script and the inline documentation in each class.
