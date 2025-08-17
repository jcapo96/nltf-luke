# NLTF-LUKE Data Analysis Framework

A comprehensive data analysis framework for processing and analyzing liquid nitrogen test facility (NLTF) data with support for multiple data formats and automated report generation.

## Overview

This framework provides a modular, extensible system for:
- Loading and converting data from various formats (Excel, CSV, JSON, etc.)
- Processing different types of signals (liquid level, H₂O concentration, temperature, purity)
- Analyzing data with configurable integration windows
- Generating comprehensive LaTeX reports
- Creating preliminary data analysis reports

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Generate Main Report**:
   ```bash
   python3 src/main.py your_config.json
   ```

2. **Generate Preliminary Report**:
   ```bash
   python3 src/preliminary_report.py your_config.json
   ```

## Configuration

The framework uses JSON configuration files that specify:
- Data file paths and converter selection
- Integration parameters and analysis preferences
- Sample information and metadata
- Report generation settings

### Configuration Guide

For detailed information about JSON configuration files, see [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).

### Quick Configuration Example

```json
{
  "Data": {
    "Path": "/path/to/data/directory",
    "Name": "dataset_prefix",
    "Converter": "SeeqNewConverter"
  },
  "Parameters": {
    "H2O": {
      "integration_time_ini": 60,
      "integration_time_end": 480,
      "offset_ini": 60,
      "offset_end": 0
    }
  }
}
```

## Key Features

### Data Format Abstraction
- **Standard Data Format**: All data is converted to a consistent internal format
- **Multiple Converters**: Support for different input file formats
- **Extensible**: Easy to add new data converters

### Analysis Capabilities
- **Integration Windows**: Configurable initial and final measurement periods
- **Multi-dataset Analysis**: Process baseline, ullage, and liquid datasets
- **Statistical Analysis**: Calculate means, standard deviations, and trends

### Report Generation
- **LaTeX Reports**: Professional-quality PDF reports with tables and plots
- **Preliminary Reports**: Raw data visualization and analysis summaries
- **Customizable Templates**: Modify report structure and content

## Architecture

The framework follows a **modular, extensible architecture** that separates concerns and makes it easy to add new functionality:

### Core Components

1. **Core Module** (`src/core/`)
   - `standard_format.py`: Standard data format definition
   - `base_classes.py`: Abstract base classes for processors and analysis

2. **Data Converters** (`src/converters/`)
   - `base_converter.py`: Abstract base class for data converters
   - `seeq_new_converter.py`: Converter for modern Seeq data format
   - `seeq_old_converter.py`: Converter for legacy Seeq data format
   - `data_format_manager.py`: Manages converter selection and registration

3. **Signal Processors** (`src/processors/`)
   - `liquid_level_processor.py`: Processes liquid level signals
   - `h2o_processor.py`: Processes H₂O concentration signals
   - `temperature_processor.py`: Processes temperature signals
   - `purity_processor.py`: Processes purity/lifetime signals

4. **Analysis Engine** (`src/analysis/`)
   - `base_analysis.py`: Abstract base class for analysis operations
   - `purity_analysis.py`: Multi-dataset purity analysis
   - `temperature_analysis.py`: Multi-dataset temperature analysis
   - `h2o_analysis.py`: Multi-dataset H₂O concentration analysis
   - `liquid_level_analysis.py`: Multi-dataset liquid level analysis
   - `main_analysis.py`: Main analysis coordinator

5. **Dataset Management** (`src/dataset/`)
   - `dataset.py`: Individual dataset handling
   - `dataset_manager.py`: Multi-dataset coordination

6. **Utilities** (`src/utils/`)
   - `data_validator.py`: Data validation utilities

7. **Main Scripts**
   - `src/main.py`: Main report generation
   - `src/preliminary_report.py`: Preliminary data analysis

## File Structure

```
NLTF-LUKE/
├── src/
│   ├── core/                   # Core data structures and base classes
│   │   ├── standard_format.py  # Standard data format definition
│   │   └── base_classes.py     # Abstract base classes
│   ├── converters/             # Data format converters
│   │   ├── base_converter.py   # Abstract converter base class
│   │   ├── seeq_new_converter.py # Modern Seeq format converter
│   │   ├── seeq_old_converter.py # Legacy Seeq format converter
│   │   └── data_format_manager.py # Converter management
│   ├── processors/             # Signal processing classes
│   │   ├── liquid_level_processor.py # Liquid level processing
│   │   ├── h2o_processor.py   # H₂O concentration processing
│   │   ├── temperature_processor.py # Temperature processing
│   │   └── purity_processor.py # Purity/lifetime processing
│   ├── analysis/               # Analysis engine
│   │   ├── base_analysis.py   # Abstract analysis base class
│   │   ├── purity_analysis.py # Purity analysis
│   │   ├── temperature_analysis.py # Temperature analysis
│   │   ├── h2o_analysis.py    # H₂O concentration analysis
│   │   ├── liquid_level_analysis.py # Liquid level analysis
│   │   └── main_analysis.py   # Main analysis coordinator
│   ├── dataset/                # Dataset management
│   │   ├── dataset.py         # Individual dataset handling
│   │   └── dataset_manager.py # Multi-dataset coordination
│   ├── utils/                  # Utility functions
│   │   └── data_validator.py  # Data validation
│   ├── main.py                 # Main report generation script
│   ├── preliminary_report.py   # Preliminary data analysis
│   └── __init__.py
├── test_data/                  # Sample data files
├── copper_tape.json           # Example configuration
├── requirements.txt            # Python dependencies
├── CONVERTER_IMPLEMENTATION.md # Guide for adding new converters
├── CONFIGURATION_GUIDE.md      # JSON configuration reference
└── README.md                  # This file
```

## Usage Examples

### Generate Main Analysis Report

```bash
# Basic usage
python3 src/main.py copper_tape.json

# The script will:
# 1. Load and convert data files
# 2. Perform analysis with specified parameters
# 3. Generate plots (purity.png, h2o_concentration.png, etc.)
# 4. Create LaTeX report (report.pdf)
```

### Generate Preliminary Report

```bash
# Create preliminary analysis
python3 src/preliminary_report.py copper_tape.json

# This generates:
# - Individual dataset plots
# - Combined signal plots
# - Summary report with integration windows
# - All plots saved to preliminary_plots/ directory
```

## Integration Parameters

The framework uses a sophisticated integration window system:

- **Initial Integration**: `integration_time_ini` minutes **before** t0=0 (run start)
- **Final Integration**: `integration_time_end` minutes **ending** at `offset_end` minutes **before** the end time

This allows for:
- Pre-run baseline measurements
- End-of-run final measurements
- Configurable measurement periods
- Consistent analysis across datasets

## Extending the Framework

The modular architecture makes it easy to extend the framework with new functionality:

### Adding New Data Converters
See [CONVERTER_IMPLEMENTATION.md](CONVERTER_IMPLEMENTATION.md) for detailed instructions on implementing new data converters.

### Adding New Signal Processors
1. Create a new processor class in `src/processors/`
2. Inherit from `BaseDataProcessor`
3. Implement required methods: `process()`, `_validate_data()`
4. Add plotting methods as needed

### Adding New Analysis Types
1. Create a new analysis class in `src/analysis/`
2. Inherit from `BaseAnalysis`
3. Implement the `analyze()` method
4. Register with the main analysis coordinator

### Adding New Signal Types
1. Extend `StandardDataFormat` in `src/core/standard_format.py`
2. Create corresponding processor and analysis classes
3. Update the dataset initialization

## Troubleshooting

### Common Issues

1. **Missing Plot Files**: Ensure all required data is available and converters are working
2. **LaTeX Compilation Errors**: Check that all plot files exist before LaTeX compilation
3. **Data Loading Issues**: Verify file paths and converter compatibility

### Debug Mode

For troubleshooting, you can temporarily add print statements to the analysis methods in the respective processor files.

## Contributing

To add new functionality:
1. Follow the existing modular structure in the appropriate directory
2. Inherit from the correct abstract base classes
3. Implement required abstract methods
4. Add appropriate error handling and validation
5. Update the relevant `__init__.py` files
6. Update documentation and add examples

## License

This project is part of the DUNE-IFIC collaboration and follows Fermilab software guidelines.