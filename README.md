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
- Data file paths
- Integration parameters
- Analysis preferences
- Converter selection

### Example Configuration Structure

```json
{
  "Data": {
    "Converter": "SeeqNewConverter",
    "Baseline": "path/to/baseline.xlsx",
    "Ullage": "path/to/ullage.xlsx",
    "Liquid": "path/to/liquid.xlsx"
  },
  "Analysis": {
    "integration_time_ini": 60,
    "integration_time_end": 480,
    "offset_ini": 60,
    "offset_end": 0
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

### Core Components

1. **Data Classes** (`src/dataClasses.py`)
   - Data processors for different signal types
   - Integration window calculations
   - Statistical analysis methods

2. **Analysis Classes** (`src/analysisClasses.py`)
   - Multi-dataset analysis coordination
   - Plot generation and management
   - Result aggregation

3. **Data Formats** (`src/dataFormats.py`)
   - Converter implementations
   - Standard data format definition
   - Format manager for converter selection

4. **Main Script** (`src/main.py`)
   - Report generation orchestration
   - LaTeX template rendering
   - Plot saving and management

5. **Preliminary Report** (`src/preliminary_report.py`)
   - Raw data visualization
   - Integration window visualization
   - Data quality assessment

## File Structure

```
NLTF-LUKE/
├── src/
│   ├── main.py                 # Main report generation script
│   ├── preliminary_report.py   # Preliminary data analysis
│   ├── dataClasses.py          # Data processing classes
│   ├── analysisClasses.py      # Analysis coordination
│   ├── dataFormats.py          # Data format converters
│   └── __init__.py
├── test_data/                  # Sample data files
├── copper_tape.json           # Example configuration
├── requirements.txt            # Python dependencies
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

## Troubleshooting

### Common Issues

1. **Missing Plot Files**: Ensure all required data is available and converters are working
2. **LaTeX Compilation Errors**: Check that all plot files exist before LaTeX compilation
3. **Data Loading Issues**: Verify file paths and converter compatibility

### Debug Mode

For troubleshooting, you can temporarily add print statements to the analysis methods in `src/dataClasses.py`.

## Contributing

To add new functionality:
1. Follow the existing class structure
2. Implement required abstract methods
3. Add appropriate error handling
4. Update documentation

## License

This project is part of the DUNE-IFIC collaboration and follows Fermilab software guidelines.