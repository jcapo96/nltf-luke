# NLTF-LUKE

## Setup Instructions

Before proceeding, ensure that you have LaTeX installed on your system, as `pdflatex` is required for some features. You can install a LaTeX distribution such as TeX Live (Linux), MacTeX (macOS), or MiKTeX (Windows).

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install Required Packages

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install <package_name>
```

### 4. Deactivate the Environment

When finished, deactivate with:

```bash
deactivate
```

## Usage

To generate a report, ensure you have a JSON file with the required entries (see the example file `copper_tape.json`). Then run:

```bash
python3 src/report_latex.py <file_name>.json
```

Replace `<file_name>.json` with your JSON file.
## JSON File Fields Explained

The JSON file used for report generation contains the following fields:

- **Author**: Information about the person preparing the report.
    - `Name`: Full name of the author.
    - `Email`: Author's email address.

- **Tester**: Details of the individual who performed the test.
    - `Name`: Full name of the tester.
    - `Email`: Tester's email address.

- **Data**: Information about the data source.
    - `Path`: Directory or location of the data files.
    - `Name`: Name identifier for the dataset.

- **Date of Receipt**: The date when the sample was received (format: MM/DD/YYYY).

- **Sample**: Description and details of the sample tested.
    - `Sample Name`: Name or identifier of the sample.
    - `Composition`: Material composition or description.
    - `Picture Location`: URL or path to images or folders containing sample pictures.
    - `Dimensions`: Physical dimensions and quantity of the sample.
    - `Source`: Origin or provider of the sample.
    - `Preparation`: Steps taken to prepare the sample for testing.

- **Results**: Summary of the test results and observations.

- **Images**: Paths to images taken before and after testing.
    - `Before`: File path to the "before" image.
    - `After`: File path to the "after" image.

- **Parameters**: Measurement or analysis parameters.
    - `H2O`: Parameters related to water measurement.
        - `manual`: Boolean indicating if manual integration is wished to be used.
        - `integration_time_ini`: Initial integration time.
        - `integration_time_end`: Final integration time.
        - `offset_ini`: Initial offset value.
        - `offset_end`: Final offset value.

Each field provides essential information for generating a comprehensive report.


