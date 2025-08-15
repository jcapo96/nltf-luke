# Implementing New Data Converters

This guide explains how to implement new data converters for the NLTF-LUKE framework. The framework uses a data format abstraction layer that allows you to support new input file formats without modifying the core analysis code.

## Overview

Data converters are responsible for:
1. **Reading** data from various file formats (Excel, CSV, JSON, etc.)
2. **Converting** the data into the framework's standard format
3. **Handling** format-specific quirks and data structures
4. **Providing** metadata about the dataset type

## Converter Architecture

### Base Class: `BaseDataConverter`

All converters inherit from `BaseDataConverter` and must implement these methods:

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd

class BaseDataConverter(ABC):
    """Abstract base class for data converters."""

    @abstractmethod
    def can_convert(self, file_path: str) -> bool:
        """Check if this converter can handle the given file."""
        pass

    @abstractmethod
    def convert(self, file_path: str) -> Optional[StandardDataFormat]:
        """Convert the file to standard format."""
        pass

    @abstractmethod
    def get_dataset_type(self, file_path: str) -> str:
        """Determine the dataset type (baseline, ullage, liquid, unknown)."""
        pass
```

### Standard Data Format

Converters must return data in the `StandardDataFormat` structure:

```python
@dataclass
class StandardDataFormat:
    """Standard format for all data in the framework."""
    dataset_name: str
    source_file: str
    timestamp: pd.Series
    liquid_level: Optional[pd.Series]
    h2o_concentration: Optional[pd.Series]
    temperature: Optional[pd.Series]
    purity: Optional[pd.Series]
```

## Implementation Steps

### Step 1: Create Converter Class

```python
from src.dataFormats import BaseDataConverter, StandardDataFormat
import pandas as pd
from typing import Optional

class MyCustomConverter(BaseDataConverter):
    """Converter for MyCustom data format."""

    def __init__(self):
        self.name = "MyCustomConverter"
        self.supported_extensions = ['.myc', '.custom']

    def can_convert(self, file_path: str) -> bool:
        """Check if file can be converted by this converter."""
        # Check file extension
        if not any(file_path.endswith(ext) for ext in self.supported_extensions):
            return False

        # Check file content (optional but recommended)
        try:
            # Try to read a small sample to verify format
            sample_data = self._read_sample(file_path)
            return self._validate_format(sample_data)
        except Exception:
            return False

    def convert(self, file_path: str) -> Optional[StandardDataFormat]:
        """Convert file to standard format."""
        try:
            # Read the data
            data_df = self._read_data(file_path)

            # Extract and process columns
            timestamp = self._extract_timestamp(data_df)
            liquid_level = self._extract_liquid_level(data_df)
            h2o_concentration = self._extract_h2o_concentration(data_df)
            temperature = self._extract_temperature(data_df)
            purity = self._extract_purity(data_df)

            # Determine dataset type
            dataset_type = self.get_dataset_type(file_path)

            # Create standard format
            return StandardDataFormat(
                dataset_name=dataset_type,
                source_file=file_path,
                timestamp=timestamp,
                liquid_level=liquid_level,
                h2o_concentration=h2o_concentration,
                temperature=temperature,
                purity=purity
            )

        except Exception as e:
            print(f"Error converting {file_path}: {e}")
            return None

    def get_dataset_type(self, file_path: str) -> str:
        """Determine dataset type from filename or content."""
        # Example logic - customize based on your needs
        filename = file_path.lower()
        if 'baseline' in filename:
            return 'baseline'
        elif 'ullage' in filename:
            return 'ullage'
        elif 'liquid' in filename:
            return 'liquid'
        else:
            return 'unknown'

    # Helper methods for data extraction
    def _read_data(self, file_path: str) -> pd.DataFrame:
        """Read data from file into DataFrame."""
        # Implement based on your file format
        # Example for CSV:
        return pd.read_csv(file_path)
        # Example for Excel:
        # return pd.read_excel(file_path)
        # Example for custom binary format:
        # return self._read_binary_format(file_path)

    def _extract_timestamp(self, data_df: pd.DataFrame) -> pd.Series:
        """Extract and convert timestamp column."""
        # Find timestamp column (customize based on your format)
        timestamp_col = None
        for col in data_df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
                timestamp_col = col
                break

        if timestamp_col is None:
            raise ValueError("No timestamp column found")

        # Convert to datetime
        timestamp = pd.to_datetime(data_df[timestamp_col])
        return timestamp

    def _extract_liquid_level(self, data_df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract liquid level data."""
        # Find liquid level column
        level_col = None
        for col in data_df.columns:
            if any(keyword in col.lower() for keyword in ['level', 'height', 'depth']):
                level_col = col
                break

        if level_col is None:
            return None

        return data_df[level_col]

    def _extract_h2o_concentration(self, data_df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract H2O concentration data."""
        # Find H2O concentration column
        h2o_col = None
        for col in data_df.columns:
            if any(keyword in col.lower() for keyword in ['h2o', 'water', 'moisture', 'humidity']):
                h2o_col = col
                break

        if h2o_col is None:
            return None

        return data_df[h2o_col]

    def _extract_temperature(self, data_df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract temperature data."""
        # Find temperature column
        temp_col = None
        for col in data_df.columns:
            if any(keyword in col.lower() for keyword in ['temp', 'temperature', 't_']):
                temp_col = col
                break

        if temp_col is None:
            return None

        return data_df[temp_col]

    def _extract_purity(self, data_df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract purity/electron lifetime data."""
        # Find purity column
        purity_col = None
        for col in data_df.columns:
            if any(keyword in col.lower() for keyword in ['purity', 'lifetime', 'electron']):
                purity_col = col
                break

        if purity_col is None:
            return None

        return data_df[purity_col]

    def _validate_format(self, sample_data) -> bool:
        """Validate that the data has the expected format."""
        # Implement validation logic
        # Example: check for required columns
        required_columns = ['timestamp', 'liquid_level']  # customize
        return all(col in sample_data.columns for col in required_columns)

    def _read_sample(self, file_path: str, n_rows: int = 5) -> pd.DataFrame:
        """Read a small sample of the file for validation."""
        # Implement based on your file format
        # Example for CSV:
        return pd.read_csv(file_path, nrows=n_rows)
```

### Step 2: Register the Converter

Add your converter to the `DataFormatManager`:

```python
# In src/dataFormats.py, add to the __init__ method:

def __init__(self):
    self.converters = [
        SeeqNewConverter(),
        SeeqOldConverter(),
        MyCustomConverter(),  # Add your converter here
        # ... other converters
    ]
```

### Step 3: Test Your Converter

Create a test script to verify your converter works:

```python
# test_my_converter.py
from src.dataFormats import DataFormatManager

def test_converter():
    format_manager = DataFormatManager()

    # Test file path
    test_file = "path/to/your/test/file.myc"

    # Check if converter can handle it
    converter = format_manager.get_converter(test_file)
    if converter:
        print(f"Found converter: {converter.name}")

        # Try to convert
        result = converter.convert(test_file)
        if result:
            print("Conversion successful!")
            print(f"Dataset: {result.dataset_name}")
            print(f"Data points: {len(result.timestamp)}")
        else:
            print("Conversion failed!")
    else:
        print("No suitable converter found")

if __name__ == "__main__":
    test_converter()
```

## Best Practices

### 1. Error Handling
- Always wrap conversion logic in try-catch blocks
- Provide meaningful error messages
- Return `None` on failure (don't raise exceptions)

### 2. Data Validation
- Validate file format before attempting conversion
- Check for required columns
- Verify data types and ranges
- Handle missing or corrupted data gracefully

### 3. Performance
- Use efficient data reading methods
- Implement lazy loading for large files if possible
- Cache validation results when appropriate

### 4. Documentation
- Document expected file format
- Provide examples of valid data
- List any dependencies or requirements

## Example: CSV Converter

Here's a complete example of a CSV converter:

```python
class CSVDataConverter(BaseDataConverter):
    """Converter for CSV files with standard column names."""

    def __init__(self):
        self.name = "CSVDataConverter"
        self.supported_extensions = ['.csv']
        self.column_mapping = {
            'timestamp': 'Time',
            'liquid_level': 'LiquidLevel_mm',
            'h2o_concentration': 'H2O_ppb',
            'temperature': 'Temperature_K',
            'purity': 'Purity_arb'
        }

    def can_convert(self, file_path: str) -> bool:
        if not file_path.endswith('.csv'):
            return False

        try:
            # Check if file can be read and has expected columns
            df = pd.read_csv(file_path, nrows=1)
            required_cols = ['Time', 'LiquidLevel_mm']
            return all(col in df.columns for col in required_cols)
        except Exception:
            return False

    def convert(self, file_path: str) -> Optional[StandardDataFormat]:
        try:
            df = pd.read_csv(file_path)

            # Extract timestamp
            timestamp = pd.to_datetime(df['Time'])

            # Extract other data
            liquid_level = df.get('LiquidLevel_mm', None)
            h2o_concentration = df.get('H2O_ppb', None)
            temperature = df.get('Temperature_K', None)
            purity = df.get('Purity_arb', None)

            dataset_type = self.get_dataset_type(file_path)

            return StandardDataFormat(
                dataset_name=dataset_type,
                source_file=file_path,
                timestamp=timestamp,
                liquid_level=liquid_level,
                h2o_concentration=h2o_concentration,
                temperature=temperature,
                purity=purity
            )

        except Exception as e:
            print(f"Error converting CSV {file_path}: {e}")
            return None

    def get_dataset_type(self, file_path: str) -> str:
        filename = file_path.lower()
        if 'baseline' in filename:
            return 'baseline'
        elif 'ullage' in filename:
            return 'ullage'
        elif 'liquid' in filename:
            return 'liquid'
        return 'unknown'
```

## Troubleshooting

### Common Issues

1. **Converter not found**: Ensure converter is registered in `DataFormatManager`
2. **Conversion fails**: Check file format and column names
3. **Data type errors**: Verify timestamp and numeric column formats
4. **Memory issues**: Use chunked reading for very large files

### Debug Tips

- Add print statements to see what's happening during conversion
- Check the data structure at each step
- Verify column names and data types
- Test with small sample files first

## Integration

Once your converter is working:

1. **Add to the framework**: Register it in `DataFormatManager`
2. **Update documentation**: Document the supported format
3. **Add tests**: Create unit tests for your converter
4. **Share**: Consider contributing back to the project

## Support

If you encounter issues implementing a converter:

1. Check the existing converter implementations for examples
2. Review the `StandardDataFormat` requirements
3. Ensure your file format is well-defined
4. Test with simple, clean data first

Remember: The goal is to make data loading as robust and user-friendly as possible!
