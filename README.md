# NLTF-LUKE Data Analysis Framework

A comprehensive data analysis framework for processing and analyzing liquid nitrogen test facility (NLTF) data with support for multiple data formats and automated report generation.

## Overview

This framework provides a modular, extensible system for:
- Loading and converting data from various formats (Excel, CSV, iHistorian, etc.)
- Processing different types of signals (liquid level, H₂O concentration, temperature, purity)
- Analyzing data with configurable integration windows
- Generating comprehensive LaTeX reports
- Creating preliminary data analysis reports

## Environment Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv luke

# Activate virtual environment
# On macOS/Linux:
source luke/bin/activate
# On Windows:
# luke\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test the framework
python3 -c "from src.converters import DataFormatManager; print('✅ Framework loaded successfully!')"
```

## Configuration Files

The framework uses JSON configuration files to specify data sources, analysis parameters, and metadata. Configuration files contain several key sections:

### Configuration File Structure

```json
{
  "Author": {
    "Name": "Your Name",
    "Email": "your.email@example.com"
  },
  "Tester": {
    "Name": "Tester Name", 
    "Email": "tester@example.com"
  },
  "Data": {
    "Path": "/path/to/data/directory",
    "Name": "dataset_prefix",
    "Converter": "iHistorianConverter"
  },
  "Date of Receipt": "MM/DD/YYYY",
  "Sample": {
    "Sample Name": "Descriptive sample name",
    "Composition": "Sample material description",
    "Picture Location": "Path to sample images",
    "Dimensions": "Sample dimensions",
    "Source": "Sample source",
    "Preparation": "Preparation procedure"
  },
  "Results": {
    "Summary": "Analysis summary text"
  },
  "Images": {
    "Before": "/path/to/before_image.jpg",
    "After": "/path/to/after_image.jpg"
  },
  "Parameters": {
    "H2O": {
      "manual": false,
      "integration_time_ini": 60,
      "integration_time_end": 480,
      "offset_ini": 60,
      "offset_end": 0
    }
  }
}
```

### Configuration Fields Explained

#### Author Section
- **Name**: Primary author of the analysis
- **Email**: Contact email for the author

#### Tester Section  
- **Name**: Person who performed the testing
- **Email**: Contact email for the tester

#### Data Section
- **Path**: Directory containing the data files
- **Name**: Prefix for dataset files (e.g., "october2025" for files like `october2025_baseline.csv`)
- **Converter**: Data converter to use:
  - `"iHistorianConverter"` - For semicolon-separated CSV files with combined signals
  - `"CsvFolderConverter"` - For CSV files organized in subdirectories
  - `"SeeqNewConverter"` - For modern Seeq Excel format
  - `"SeeqOldConverter"` - For legacy Seeq Excel format

#### Sample Section
- **Sample Name**: Descriptive name for the test sample
- **Composition**: Material composition description
- **Picture Location**: Path to sample images
- **Dimensions**: Physical dimensions of the sample
- **Source**: Where the sample was obtained
- **Preparation**: Sample preparation procedure

#### Results Section
- **Summary**: Text summary of the analysis results

#### Images Section
- **Before**: Path to "before test" sample image
- **After**: Path to "after test" sample image

#### Parameters Section
- **H2O.manual**: Whether to use manual time selection (true/false)
- **H2O.integration_time_ini**: Initial integration window duration (minutes)
- **H2O.integration_time_end**: Final integration window duration (minutes)  
- **H2O.offset_ini**: Initial offset from start time (minutes)
- **H2O.offset_end**: Final offset from end time (minutes)

## Usage

### 1. Generate Main Analysis Report

The main analysis generates a comprehensive LaTeX PDF report with statistical analysis and plots.

```bash
# Activate virtual environment
source luke/bin/activate

# Run main analysis
python3 src/main.py your_config.json
```

**Output files:**
- `report.pdf` - Main LaTeX report
- `purity.png` - Purity analysis plot
- `h2o_concentration.png` - H₂O concentration plot  
- `temperature.png` - Temperature analysis plot
- `level.png` - Liquid level plot
- `report.tex` - LaTeX source file

### 2. Generate Preliminary Analysis

The preliminary analysis creates raw data plots and summary statistics for initial data inspection.

```bash
# Activate virtual environment
source luke/bin/activate

# Run preliminary analysis
python3 src/preliminary_report.py your_config.json
```

**Output files:**
- `preliminary_plots/` directory containing:
  - Individual signal plots for each dataset (baseline, ullage, liquid)
  - Combined plots showing all signals together
  - `preliminary_summary.txt` - Detailed data summary

### Example Workflow

```bash
# 1. Set up environment
python3 -m venv luke
source luke/bin/activate
pip install -r requirements.txt

# 2. Prepare your data and configuration file
# Edit your_config.json with appropriate paths and parameters

# 3. Run preliminary analysis first
python3 src/preliminary_report.py your_config.json

# 4. Review preliminary plots and adjust parameters if needed

# 5. Run main analysis
python3 src/main.py your_config.json

# 6. Check generated report.pdf
```

## Supported Data Formats

The framework supports multiple data input formats through its converter system:

### iHistorian Format
- **File Type**: Semicolon-separated CSV files
- **Structure**: All signals combined in single file with sparse data
- **Files**: `*_baseline.csv`, `*_liquid.csv`, `*_ullage.csv`
- **Converter**: `iHistorianConverter`
- **Use Case**: Modern data acquisition systems with combined signal output

### CSV Folder Format  
- **File Type**: CSV files organized in subdirectories
- **Structure**: Separate files per signal type in baseline/ullage/liquid folders
- **Files**: `ae_*.csv` (H₂O), `lt_*.csv` (level), `prm_*.csv` (purity), `te_*.csv` (temperature)
- **Converter**: `CsvFolderConverter`
- **Use Case**: Legacy data organization with separate signal files

### Seeq Formats
- **File Type**: Excel files (.xlsx/.xls)
- **Structure**: Modern and legacy Seeq data export formats
- **Converters**: `SeeqNewConverter`, `SeeqOldConverter`
- **Use Case**: Seeq data analysis platform exports

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
   - `csv_folder_converter.py`: Converter for CSV files in subdirectories
   - `ihistorian_converter.py`: Converter for iHistorian semicolon-separated CSV format
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
│   │   ├── csv_folder_converter.py # CSV subdirectory converter
│   │   ├── ihistorian_converter.py # iHistorian CSV converter
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

1. **Module Not Found Errors**: 
   - Ensure virtual environment is activated: `source luke/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`

2. **Data Loading Issues**:
   - Verify file paths in configuration file are correct
   - Check that data files exist and are readable
   - Ensure correct converter is specified for your data format

3. **Missing Plot Files**: 
   - Ensure all required data is available and converters are working
   - Check that datasets contain the expected signal types

4. **LaTeX Compilation Errors**: 
   - Check that all plot files exist before LaTeX compilation
   - Ensure LaTeX is installed: `brew install --cask mactex` (macOS) or `sudo apt-get install texlive-full` (Ubuntu)

5. **Converter Not Found**:
   - Verify the converter name in configuration file matches available converters
   - Check that the data format is supported by the specified converter

### Debug Mode

For troubleshooting, you can temporarily add print statements to the analysis methods in the respective processor files.

### Getting Help

1. Check the configuration file format against the examples in this README
2. Verify your data files match the expected format for the chosen converter
3. Run preliminary analysis first to check data loading before main analysis
4. Check the generated `preliminary_summary.txt` for data statistics and issues

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