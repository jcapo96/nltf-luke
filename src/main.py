import json
from jinja2 import Template
import subprocess
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysisClasses import Analysis, DatasetManager
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")

if len(sys.argv) != 2:
    print(f"Usage: python {os.path.basename(__file__)} <input.json>")
    sys.exit(1)

json_file = sys.argv[1]
with open(json_file) as f:
    data = json.load(f)

# Create simplified variables
sample = data["Sample"]
results = data["Results"]
images = data["Images"]
parameters = data["Parameters"]
h2o_parameters = parameters["H2O"]

if "signal" in h2o_parameters:
    if h2o_parameters["signal"] == "NEW":
        signal_name = "PAB_S1_AE_611_AR_REAL_F_CV"
    elif h2o_parameters["signal"] == "OLD":
        signal_name = "PAB_S1_AE_600_AR_REAL_F_CV_OLD"
    else:
        signal_name = h2o_parameters["signal"]
elif "signal" not in h2o_parameters:
    signal_name = "PAB_S1_AE_611_AR_REAL_F_CV"

# Create dataset paths dictionary for the new DatasetManager
data_path = data["Data"]["Path"]
data_name = data["Data"]["Name"]
dataset_paths = {
    'baseline': f"{data_path}/{data_name}_baseline.xlsx",
    'ullage': f"{data_path}/{data_name}_ullage.xlsx",
    'liquid': f"{data_path}/{data_name}_liquid.xlsx"
}

# Initialize the new DatasetManager
dataset_manager = DatasetManager(dataset_paths, data_name, json_file)

# Create analysis object using the new structure
analysis = Analysis(dataset_manager)

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Run analyses and get results
purity = analysis.purity(show=False, ax=ax1, manual=False,
    integration_time_ini=h2o_parameters["integration_time_ini"], integration_time_end=h2o_parameters["integration_time_end"],
    offset_ini=h2o_parameters["offset_ini"], offset_end=h2o_parameters["offset_end"])

h2o_concentration = analysis.h2oConcentration(show=False, ax=ax2, manual=h2o_parameters["manual"],
    integration_time_ini=h2o_parameters["integration_time_ini"], integration_time_end=h2o_parameters["integration_time_end"],
    offset_ini=h2o_parameters["offset_ini"], offset_end=h2o_parameters["offset_end"])

temperature = analysis.temperature(show=False, ax=ax3, manual=False,
    integration_time_ini=h2o_parameters["integration_time_ini"], integration_time_end=h2o_parameters["integration_time_end"],
    offset_ini=h2o_parameters["offset_ini"], offset_end=h2o_parameters["offset_end"])

level = analysis.level(ax=ax4, manual=False)

# Save plots
fig1.savefig("purity.png", dpi=300)
fig2.savefig("h2o_concentration.png", dpi=300)
fig3.savefig("temperature.png", dpi=300)
fig4.savefig("level.png", dpi=300)

# Get analysis results for template rendering
analysis_results = analysis.get_analysis_results()

# Fill template
with open("report_template.tex") as f:
    template = Template(f.read())

def format_datetime(dt):
    """Format datetime for display, handling 'N/A' and other edge cases."""
    if dt == 'N/A' or dt is None:
        return 'N/A'
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return 'N/A'
    return dt.strftime("%m/%d/%y %H:%M")

# Replace '%' with '\%' in relevant fields if present
def escape_percent(s):
    return s.replace('%', '\\%') if '%' in s else s

# Helper function to extract data from analysis results
def get_dataset_data(analysis_results, dataset_type, data_type):
    """Extract data for a specific dataset and data type from analysis results."""
    if dataset_type in analysis_results.get(data_type, {}):
        return analysis_results[data_type][dataset_type]
    return None

# Helper function to extract H2O data from the nested structure
def extract_h2o_data(h2o_result):
    """Extract H2O concentration data from the analysis result structure."""
    if h2o_result and 'h2o_data' in h2o_result and h2o_result['h2o_data'] is not None:
        h2o_data = h2o_result['h2o_data']
        # Convert to the format expected by the template
        return {
            'initial': h2o_data.get('h2o_ini', 0),
            'initial_error': h2o_data.get('h2o_ini_err', 0),
            'final': h2o_data.get('h2o_end', 0),
            'final_error': h2o_data.get('h2o_end_err', 0)
        }
    # Return default values if no H2O data available
    return {
        'initial': 0,
        'initial_error': 0,
        'final': 0,
        'final_error': 0
    }

# Helper function to extract temperature data from the nested structure
def extract_temp_data(temp_result):
    """Extract temperature data from the analysis result structure."""
    if temp_result and 'temp_data' in temp_result and temp_result['temp_data'] is not None:
        temp_data = temp_result['temp_data']
        # Convert to the format expected by the template
        return {
            'initial': temp_data.get('temp_ini', 0),
            'initial_error': temp_data.get('temp_ini_err', 0),
            'final': temp_data.get('temp_end', 0),
            'final_error': temp_data.get('temp_end_err', 0)
        }
    # Return default values if no temperature data available
    return {
        'initial': 0,
        'initial_error': 0,
        'final': 0,
        'final_error': 0
    }

# Extract baseline data
baseline_h2o = get_dataset_data(analysis_results, 'baseline', 'h2o_concentration')
baseline_temp = get_dataset_data(analysis_results, 'baseline', 'temperature')
baseline_level = get_dataset_data(analysis_results, 'baseline', 'liquid_level')

# Extract ullage data
ullage_h2o = get_dataset_data(analysis_results, 'ullage', 'h2o_concentration')
ullage_temp = get_dataset_data(analysis_results, 'ullage', 'temperature')
ullage_level = get_dataset_data(analysis_results, 'ullage', 'liquid_level')

# Extract liquid data
liquid_h2o = get_dataset_data(analysis_results, 'liquid', 'h2o_concentration')
liquid_temp = get_dataset_data(analysis_results, 'liquid', 'temperature')
liquid_level = get_dataset_data(analysis_results, 'liquid', 'liquid_level')

# Extract the actual data from the nested structure
baseline_h2o_data = extract_h2o_data(baseline_h2o)
ullage_h2o_data = extract_h2o_data(ullage_h2o)
liquid_h2o_data = extract_h2o_data(liquid_h2o)

baseline_temp_data = extract_temp_data(baseline_temp)
ullage_temp_data = extract_temp_data(ullage_temp)
liquid_temp_data = extract_temp_data(liquid_temp)

# Helper function to safely get values with defaults
def safe_get(data, key, default=0):
    """Safely get a value from data with a default fallback."""
    if data and key in data:
        return data[key]
    return default

# Check if we have any data to work with
if not any([baseline_h2o_data, ullage_h2o_data, liquid_h2o_data]):
    baseline_h2o_data = {'initial': 0, 'initial_error': 0, 'final': 0, 'final_error': 0}
    ullage_h2o_data = {'initial': 0, 'initial_error': 0, 'final': 0, 'final_error': 0}
    liquid_h2o_data = {'initial': 0, 'initial_error': 0, 'final': 0, 'final_error': 0}

if not any([baseline_temp_data, ullage_temp_data, liquid_temp_data]):
    baseline_temp_data = {'initial': 0, 'initial_error': 0, 'final': 0, 'final_error': 0}
    ullage_temp_data = {'initial': 0, 'initial_error': 0, 'final': 0, 'final_error': 0}
    liquid_temp_data = {'initial': 0, 'initial_error': 0, 'final': 0, 'final_error': 0}

if not any([baseline_level, ullage_level, liquid_level]):
    baseline_level = {'start_time': 'N/A', 'end_time': 'N/A'}
    ullage_level = {'start_time': 'N/A', 'end_time': 'N/A'}
    liquid_level = {'start_time': 'N/A', 'end_time': 'N/A'}

rendered = template.render(
    author={
        "name": data["Author"]["Name"],
        "email": data["Author"]["Email"]
    },
    tester={
        "name": data["Tester"]["Name"],
        "email": data["Tester"]["Email"]
    },
    date=data["Date of Receipt"],
    sample={
        "name": sample["Sample Name"],
        "composition": escape_percent(sample["Composition"]),
        "picture": sample["Picture Location"],
        "dimensions": sample["Dimensions"],
        "source": sample["Source"],
        "preparation": escape_percent(sample["Preparation"]),
    },
    baseline={
        "start_date": format_datetime(safe_get(baseline_level, 'start_time', 'N/A')),
        "end_date": format_datetime(safe_get(baseline_level, 'end_time', 'N/A')),
        "initial_concentration": round(safe_get(baseline_h2o_data, 'initial', 0), 2),
        "final_concentration": round(safe_get(baseline_h2o_data, 'final', 0), 1),
        "concentration": round(safe_get(baseline_h2o_data, 'final', 0) - safe_get(baseline_h2o_data, 'initial', 0), 1),
        "initial_concentration_err": round(safe_get(baseline_h2o_data, 'initial_error', 0), 2),
        "final_concentration_err": round(safe_get(baseline_h2o_data, 'final_error', 0), 1),
        "concentration_err": round(np.sqrt(safe_get(baseline_h2o_data, 'final_error', 0)**2 + safe_get(baseline_h2o_data, 'initial_error', 0)**2), 1)
    },
    ullage={
        "start_date": format_datetime(safe_get(ullage_level, 'start_time', 'N/A')),
        "end_date": format_datetime(safe_get(ullage_level, 'end_time', 'N/A')),
        "initial_concentration": round(safe_get(ullage_h2o_data, 'initial', 0), 2),
        "final_concentration": round(safe_get(ullage_h2o_data, 'final', 0), 1),
        "concentration": round(safe_get(ullage_h2o_data, 'final', 0) - safe_get(ullage_h2o_data, 'initial', 0), 1),
        "initial_concentration_err": round(safe_get(ullage_h2o_data, 'initial_error', 0), 2),
        "final_concentration_err": round(safe_get(ullage_h2o_data, 'final_error', 0), 1),
        "concentration_err": round(np.sqrt(safe_get(ullage_h2o_data, 'final_error', 0)**2 + safe_get(ullage_h2o_data, 'initial_error', 0)**2), 1),
        "temperature": round(safe_get(ullage_temp_data, 'final', 0), 1),
        "temperature_err": round(safe_get(ullage_temp_data, 'final_error', 0), 1)
    },
    liquid={
        "start_date": format_datetime(safe_get(liquid_level, 'start_time', 'N/A')),
        "end_date": format_datetime(safe_get(liquid_level, 'end_time', 'N/A')),
        "initial_concentration": round(safe_get(liquid_h2o_data, 'initial', 0), 2),
        "final_concentration": round(safe_get(liquid_h2o_data, 'final', 0), 1),
        "concentration_err": round(np.sqrt(safe_get(liquid_h2o_data, 'final_error', 0)**2 + safe_get(liquid_h2o_data, 'initial_error', 0)**2), 1),
        "concentration": round(safe_get(liquid_h2o_data, 'final', 0) - safe_get(liquid_h2o_data, 'initial', 0), 1),
        "initial_concentration_err": round(safe_get(liquid_h2o_data, 'initial_error', 0), 2),
        "final_concentration_err": round(safe_get(liquid_h2o_data, 'final_error', 0), 1),
        "temperature": round(safe_get(liquid_temp_data, 'final', 0), 1),
        "temperature_err": round(safe_get(liquid_temp_data, 'final_error', 0), 1)
    },
    results={
        "summary": results["Summary"],
        "purity_img": "purity.png",
        "h2o_img": "h2o_concentration.png",
        "temperature_img": "temperature.png",
        "level_img": "level.png"
    },
    parameters={
        "h2o_parameters": {
            "integration_time_ini": round(h2o_parameters["integration_time_ini"], 0),
            "integration_time_end": h2o_parameters["integration_time_end"],
            "offset_ini": round(h2o_parameters["offset_ini"], 0),
            "offset_end": h2o_parameters["offset_end"]/60
        }
    },
    images={
        "before": images["Before"].strip(),
        "after": images["After"].strip(),
    }
)

# Write to .tex file
with open("report.tex", "w") as f:
    f.write(rendered)

# Compile PDF
subprocess.run(["pdflatex", "report.tex"])

print("✅ PDF report generated: report.pdf")

for fname in ["purity.png", "h2o_concentration.png", "temperature.png", "level.png"]:
    if os.path.exists(fname):
        os.remove(fname)
