# JSON Configuration Guide

This guide explains the complete structure and required fields for JSON configuration files used by the NLTF-LUKE framework.

## Overview

The framework uses JSON configuration files to specify all parameters needed for data analysis and report generation. These files contain information about the sample, data sources, analysis parameters, and report metadata.

## Complete Configuration Structure

```json
{
    "Author": {
        "Name": "Your Name",
        "Email": "your.email@institution.edu"
    },
    "Tester": {
        "Name": "Tester Name",
        "Email": "tester.email@institution.edu"
    },
    "Data": {
        "Path": "/path/to/data/directory",
        "Name": "dataset_prefix",
        "Converter": "SeeqNewConverter"
    },
    "Date of Receipt": "MM/DD/YYYY",
    "Sample": {
        "Sample Name": "Sample Description",
        "Composition": "Material composition details",
        "Picture Location": "Path or URL to sample pictures",
        "Dimensions": "Physical dimensions",
        "Source": "Sample origin",
        "Preparation": "Sample preparation steps"
    },
    "Results": {
        "Summary": "Analysis results summary"
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

## Required Fields

### 1. **Author Section** (Required)
```json
"Author": {
    "Name": "Your Name",
    "Email": "your.email@institution.edu"
}
```
- **Name**: Full name of the person conducting the analysis
- **Email**: Contact email for the author

### 2. **Tester Section** (Required)
```json
"Tester": {
    "Name": "Tester Name",
    "Email": "tester.email@institution.edu"
}
```
- **Name**: Full name of the person who tested the sample
- **Email**: Contact email for the tester

### 3. **Data Section** (Required)
```json
"Data": {
    "Path": "/path/to/data/directory",
    "Name": "dataset_prefix",
    "Converter": "SeeqNewConverter"
}
```
- **Path**: Directory containing the data files
- **Name**: Prefix for the dataset files (e.g., "copper_tape" for "copper_tape_baseline.xlsx")
- **Converter**: Data converter to use (see available converters below)

#### **Available Converters:**
- `"SeeqNewConverter"`: Modern Seeq data format (default)
- `"SeeqOldConverter"`: Legacy Seeq data format
- Custom converters can be added to the framework

#### **Expected Data Files:**
The framework expects these files in the specified data directory:
- `{dataset_prefix}_baseline.xlsx` - Baseline (no sample) data
- `{dataset_prefix}_ullage.xlsx` - Ullage (gas phase) data
- `{dataset_prefix}_liquid.xlsx` - Liquid (submerged) data

### 4. **Date of Receipt** (Required)
```json
"Date of Receipt": "MM/DD/YYYY"
```
- Date when the sample was received for testing
- Format: MM/DD/YYYY (e.g., "05/17/2025")

### 5. **Sample Section** (Required)
```json
"Sample": {
    "Sample Name": "Sample Description",
    "Composition": "Material composition details",
    "Picture Location": "Path or URL to sample pictures",
    "Dimensions": "Physical dimensions",
    "Source": "Sample origin",
    "Preparation": "Sample preparation steps"
}
```
- **Sample Name**: Descriptive name of the sample
- **Composition**: Material composition and properties
- **Picture Location**: Path to sample pictures or URL
- **Dimensions**: Physical dimensions and measurements
- **Source**: Origin of the sample (person/institution)
- **Preparation**: Steps taken to prepare the sample for testing

### 6. **Results Section** (Required)
```json
"Results": {
    "Summary": "Analysis results summary"
}
```
- **Summary**: Text description of the analysis results and conclusions

### 7. **Images Section** (Required)
```json
"Images": {
    "Before": "/path/to/before_image.jpg",
    "After": "/path/to/after_image.jpg"
}
```
- **Before**: Path to image of the sample before testing
- **After**: Path to image of the sample after testing
- Supported formats: JPG, JPEG, PNG
- Can be absolute paths or relative to the configuration file

### 8. **Parameters Section** (Required)
```json
"Parameters": {
    "H2O": {
        "manual": false,
        "integration_time_ini": 60,
        "integration_time_end": 480,
        "offset_ini": 60,
        "offset_end": 0
    }
}
```

#### **H2O Parameters:**
- **manual**: Whether to use manual time selection (boolean)
- **integration_time_ini**: Initial integration time in **minutes** before run start
- **integration_time_end**: Final integration time in **minutes** before run end
- **offset_ini**: Initial offset in **minutes** from run start
- **offset_end**: Final offset in **minutes** from run end

## Configuration Examples

### **Example 1: Copper Tape Sample**
```json
{
    "Author": {
        "Name": "Jordi Capó",
        "Email": "jordi.capo@ific.uv.es"
    },
    "Tester": {
        "Name": "Flor de Maria Blaszczyk",
        "Email": "fblaszcz@fnal.gov"
    },
    "Data": {
        "Path": "/Users/jcapo/cernbox/NLTFdata/COPPER_TAPE",
        "Name": "copper_tape",
        "Converter": "SeeqNewConverter"
    },
    "Date of Receipt": "05/17/2025",
    "Sample": {
        "Sample Name": "Copper Foil Tape",
        "Composition": "3M 1181 Copper Foil tape with conductive adhesive",
        "Picture Location": "https://drive.google.com/drive/folders/1NWR9tC9DTZSDlbT2mzevMrnqrA-BBfu_?usp=sharing",
        "Dimensions": "11 equal stripes of 20 cm by 3 cm each: 660 cm²",
        "Source": "Philippe Rosier (IJCLab)",
        "Preparation": "Sectioned into 11 uniform pieces (20 by 3 cm), folded and glued onto themselves onto a stainless steel wire to form single strips (10 by 3 cm), and cleaned with 200-proof ethanol."
    },
    "Results": {
        "Summary": "No significant water outgassing observed. The analysis on the electron lifetime shows the addition of the sample didn't have an impact on it."
    },
    "Images": {
        "Before": "/Users/jcapo/cernbox/NLTFdata/COPPER_TAPE/IMAGES/copper_tape_before_A.jpeg",
        "After": "/Users/jcapo/cernbox/NLTFdata/COPPER_TAPE/IMAGES/copper_tape_after_A.jpeg"
    },
    "Parameters": {
        "H2O": {
            "manual": false,
            "integration_time_ini": 60,
            "offset_ini": 60,
            "integration_time_end": 480,
            "offset_end": 0
        }
    }
}
```

### **Example 2: ABS Polymer Sample**
```json
{
    "Author": {
        "Name": "Jordi Capó",
        "Email": "jordi.capo@ific.uv.es"
    },
    "Tester": {
        "Name": "Flor de Maria Blaszczyk",
        "Email": "fblaszcz@fnal.gov"
    },
    "Data": {
        "Path": "/Users/jcapo/cernbox/NLTFdata/ABS",
        "Name": "abs",
        "Converter": "SeeqNewConverter"
    },
    "Date of Receipt": "03/06/2025",
    "Sample": {
        "Sample Name": "ABS (Thermoplastic polymer)",
        "Composition": "Acrylonitrile Butadiene Styrene",
        "Picture Location": "See picture tab",
        "Dimensions": "2 samples of 2.5 in. x 2.5 in. x 1/4 in.",
        "Source": "Jon Urheim (Indiana University)",
        "Preparation": "Cleaned with Ethyl alcohol 200 proof"
    },
    "Results": {
        "Summary": "Water outgassing occurs when the sample is in the ullage. No visible impact on electron lifetime. No visible physical damage on the sample."
    },
    "Images": {
        "Before": "/Users/jcapo/cernbox/NLTFdata/ABS/IMAGES/ABS_before1.jpg",
        "After": "/Users/jcapo/cernbox/NLTFdata/ABS/IMAGES/ABS_after1.jpg"
    },
    "Parameters": {
        "H2O": {
            "manual": false,
            "integration_time_ini": 60,
            "offset_ini": 60,
            "integration_time_end": 480,
            "offset_end": 0
        }
    }
}
```

## Integration Time Parameters Explained

### **Time Window Logic:**
The framework uses a sophisticated integration window system:

1. **Initial Integration Window**:
   - **Start**: `integration_time_ini` minutes **before** t0=0 (run start)
   - **End**: `offset_ini` minutes **after** t0=0 (run start)
   - **Purpose**: Measure baseline conditions before the sample is introduced

2. **Final Integration Window**:
   - **Start**: `offset_end` minutes **before** the end of the run
   - **End**: `integration_time_end` minutes **before** the end of the run
   - **Purpose**: Measure final conditions after the sample has been exposed

### **Example Timeline:**
```
Time (minutes):  -60    0      +60    +420   +480
                |      |      |      |      |
                |<---->|      |      |<---->|
                | ini  | RUN  |      | end  |
                | win  |      |      | win  |
```

- **Initial window**: -60 to +60 minutes (120 minutes total)
- **Run duration**: 0 to +480 minutes (8 hours)
- **Final window**: +420 to +480 minutes (60 minutes total)

## Validation and Troubleshooting

### **Common Configuration Errors:**

1. **Missing Required Fields**:
   - All sections marked as "Required" must be present
   - Missing fields will cause the framework to fail

2. **Invalid File Paths**:
   - Ensure data directory exists and contains required files
   - Check that image paths are correct and accessible

3. **Invalid Date Format**:
   - Use MM/DD/YYYY format (e.g., "05/17/2025")
   - Don't use ISO format or other date representations

4. **Invalid Converter Name**:
   - Use exact converter names: "SeeqNewConverter" or "SeeqOldConverter"
   - Check for typos and case sensitivity

### **Data File Requirements:**

1. **File Naming Convention**:
   - Must follow pattern: `{dataset_prefix}_{type}.xlsx`
   - Types: `baseline`, `ullage`, `liquid`
   - Example: `copper_tape_baseline.xlsx`

2. **File Format**:
   - Excel files (.xlsx format)
   - Must contain timestamp and signal columns
   - Compatible with the specified converter

3. **Data Quality**:
   - Timestamps must be in chronological order
   - Signal data should not contain excessive gaps
   - Liquid level data is required for time window detection

### **Testing Your Configuration:**

1. **Validate JSON Syntax**:
   ```bash
   python3 -m json.tool your_config.json
   ```

2. **Check File Paths**:
   ```bash
   ls -la /path/to/data/directory/
   ls -la /path/to/image/before.jpg
   ls -la /path/to/image/after.jpg
   ```

3. **Test with Framework**:
   ```bash
   python3 src/preliminary_report.py your_config.json
   ```

## Advanced Configuration

### **Custom Converters:**
If you're using a custom data converter:

1. **Ensure the converter is registered** in the framework
2. **Use the exact converter name** in the configuration
3. **Verify data format compatibility** with your converter

### **Multiple Sample Configurations:**
For testing multiple samples:

1. **Create separate configuration files** for each sample
2. **Use descriptive names** (e.g., `copper_tape.json`, `abs_polymer.json`)
3. **Keep data directories organized** by sample type

### **Parameter Optimization:**
The integration parameters can be optimized based on:

1. **Sample characteristics** (material type, size, preparation)
2. **Experimental conditions** (temperature, pressure, duration)
3. **Data quality** (sampling rate, noise levels)

## Support and Troubleshooting

### **Getting Help:**
1. **Check the main README.md** for framework overview
2. **Review CONVERTER_IMPLEMENTATION.md** for converter details
3. **Examine example configurations** in the project directory
4. **Validate your JSON syntax** using online tools

### **Common Issues:**
- **"Configuration file not found"**: Check file path and permissions
- **"No converter found"**: Verify converter name and registration
- **"Data loading failed"**: Check data file paths and formats
- **"Missing required data"**: Ensure all three dataset types are present

### **Best Practices:**
1. **Use absolute paths** for data and image files
2. **Test with preliminary report** before generating main report
3. **Keep backup copies** of working configurations
4. **Document any custom parameters** or special requirements
5. **Validate data files** before running analysis
