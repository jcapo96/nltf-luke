#!/usr/bin/env python3
"""
Main script for generating LaTeX reports from NLTF-LUKE data analysis.
"""

import sys
import os
import json
import warnings
import numpy as np
from datetime import datetime
from jinja2 import Template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress OpenPyXL warnings about default styles
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import Analysis
from dataset import DatasetManager
from converters import DataFormatManager

plt.style.use("style.mplstyle")

if len(sys.argv) != 2:
    print("Usage: python3 src/main.py <json_file>")
    sys.exit(1)

json_file = sys.argv[1]

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file '{json_file}' not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in '{json_file}'.")
    sys.exit(1)

data_path = data["Data"]["Path"]
data_name = data["Data"]["Name"]

dataset_paths = {
    'baseline': os.path.join(data_path, f"{data_name}_baseline.xlsx"),
    'ullage': os.path.join(data_path, f"{data_name}_ullage.xlsx"),
    'liquid': os.path.join(data_path, f"{data_name}_liquid.xlsx")
}

print(f"Data path: {data_path}")
print(f"Dataset name: {data_name}")
print(f"Dataset paths:")
for dataset_type, path in dataset_paths.items():
    print(f"   {dataset_type}: {path}")

sample = data["Sample"]
results = data["Results"]
images = data["Images"]

dataset_manager = DatasetManager(dataset_paths, data_name, json_file)
analysis = Analysis(dataset_manager)

print("Starting NLTF-LUKE Report Generation")
print("=" * 50)

h2o_parameters = data["Parameters"]["H2O"]

print("Setting up analysis plots...")
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig2, ax2 = plt.subplots(figsize=(8, 5))
fig3, ax3 = plt.subplots(figsize=(8, 5))
fig4, ax4 = plt.subplots(figsize=(8, 5))

print("Running data analysis...")
with tqdm(total=4, desc="Analysis Progress", unit="analysis") as pbar:
    purity = analysis.purity(show=False, ax=ax1, manual=False,
        integration_time_ini=h2o_parameters["integration_time_ini"],
        integration_time_end=h2o_parameters["integration_time_end"],
        offset_ini=h2o_parameters["offset_ini"],
        offset_end=h2o_parameters["offset_end"])
    pbar.update(1)

    h2o_concentration = analysis.h2oConcentration(show=False, ax=ax2, manual=h2o_parameters["manual"],
        integration_time_ini=h2o_parameters["integration_time_ini"],
        integration_time_end=h2o_parameters["integration_time_end"],
        offset_ini=h2o_parameters["offset_ini"],
        offset_end=h2o_parameters["offset_end"])
    pbar.update(1)

    temperature = analysis.temperature(show=False, ax=ax3, manual=False,
        integration_time_ini=h2o_parameters["integration_time_ini"],
        integration_time_end=h2o_parameters["integration_time_end"],
        offset_ini=h2o_parameters["offset_ini"],
        offset_end=h2o_parameters["offset_end"])
    pbar.update(1)

    level = analysis.level(ax=ax4, manual=False)
    pbar.update(1)

print("Saving analysis plots...")
with tqdm(total=4, desc="Saving Plots", unit="plot") as pbar:
    fig1.savefig("purity.png", dpi=150, format='png', facecolor='white', edgecolor='none')
    pbar.update(1)

    fig2.savefig("h2o_concentration.png", dpi=150, format='png', facecolor='white', edgecolor='none')
    pbar.update(1)

    fig3.savefig("temperature.png", dpi=150, format='png', facecolor='white', edgecolor='none')
    pbar.update(1)

    fig4.savefig("level.png", dpi=150, format='png', facecolor='white', edgecolor='none')
    pbar.update(1)

print("Preparing report data...")
analysis_results = analysis.get_analysis_results()

print("Loading LaTeX template...")
with open("report_template.tex") as f:
    template = Template(f.read())

def format_datetime(dt):
    if dt == 'N/A' or dt is None:
        return 'N/A'
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return 'N/A'
    return dt.strftime("%m/%d/%y %H:%M")

def escape_percent(s):
    return s.replace('%', '\\%') if '%' in s else s

def get_dataset_data(analysis_results, dataset_type, data_type):
    if dataset_type in analysis_results.get(data_type, {}):
        return analysis_results[data_type][dataset_type]
    return None

def extract_h2o_data(h2o_result):
    if h2o_result and 'h2o_data' in h2o_result and h2o_result['h2o_data'] is not None:
        h2o_data = h2o_result['h2o_data']
        return {
            'initial': h2o_data.get('h2o_ini', 0),
            'initial_error': h2o_data.get('h2o_ini_err', 0),
            'final': h2o_data.get('h2o_end', 0),
            'final_error': h2o_data.get('h2o_end_err', 0)
        }
    return {
        'initial': 0,
        'initial_error': 0,
        'final': 0,
        'final_error': 0
    }

def extract_temp_data(temp_result):
    if temp_result and 'temp_data' in temp_result and temp_result['temp_data'] is not None:
        temp_data = temp_result['temp_data']
        return {
            'initial': temp_data.get('temp_ini', 0),
            'initial_error': temp_data.get('temp_ini_err', 0),
            'final': temp_data.get('temp_end', 0),
            'final_error': temp_data.get('temp_end_err', 0)
        }
    return {
        'initial': 0,
        'initial_error': 0,
        'final': 0,
        'final_error': 0
    }

print("Extracting analysis data...")
print(f"Analysis results keys: {list(analysis_results.keys())}")
for key, value in analysis_results.items():
    print(f"  {key}: {list(value.keys()) if isinstance(value, dict) else type(value)}")

baseline_h2o = get_dataset_data(analysis_results, 'baseline', 'h2o_concentration')
baseline_temp = get_dataset_data(analysis_results, 'baseline', 'temperature')
baseline_level = get_dataset_data(analysis_results, 'baseline', 'liquid_level')

ullage_h2o = get_dataset_data(analysis_results, 'ullage', 'h2o_concentration')
ullage_temp = get_dataset_data(analysis_results, 'ullage', 'temperature')
ullage_level = get_dataset_data(analysis_results, 'ullage', 'liquid_level')

liquid_h2o = get_dataset_data(analysis_results, 'liquid', 'h2o_concentration')
liquid_temp = get_dataset_data(analysis_results, 'liquid', 'temperature')
liquid_level = get_dataset_data(analysis_results, 'liquid', 'liquid_level')

baseline_h2o_data = extract_h2o_data(baseline_h2o)
ullage_h2o_data = extract_h2o_data(ullage_h2o)
liquid_h2o_data = extract_h2o_data(liquid_h2o)

baseline_temp_data = extract_temp_data(baseline_temp)
ullage_temp_data = extract_temp_data(ullage_temp)
liquid_temp_data = extract_temp_data(liquid_temp)

def safe_get(data, key, default=0):
    if data and key in data:
        return data[key]
    return default

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

print("Rendering LaTeX template...")
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
        "concentration": round(safe_get(liquid_h2o_data, 'final', 0) - safe_get(liquid_h2o_data, 'initial', 0), 1),
        "initial_concentration_err": round(safe_get(liquid_h2o_data, 'initial_error', 0), 2),
        "final_concentration_err": round(safe_get(liquid_h2o_data, 'final_error', 0), 1),
        "concentration_err": round(np.sqrt(safe_get(liquid_h2o_data, 'final_error', 0)**2 + safe_get(liquid_h2o_data, 'initial_error', 0)**2), 1),
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
            "integration_time_end": round(h2o_parameters["integration_time_end"]/60, 0),
            "offset_ini": round(h2o_parameters["offset_ini"], 0),
            "offset_end": round(h2o_parameters["offset_end"]/60, 0)
        }
    },
    images={
        "before": images["Before"],
        "after": images["After"]
    }
)

print("Writing LaTeX file...")
with open("report.tex", "w") as f:
    f.write(rendered)

print("Compiling LaTeX to PDF...")
try:
    import subprocess

    print("  First compilation pass...")
    result1 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"],
                           capture_output=True, text=True)

    print("  Second compilation pass (resolving references)...")
    result2 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"],
                           capture_output=True, text=True)

    if os.path.exists("report.pdf") and os.path.getsize("report.pdf") > 0:
        print("PDF report generated: report.pdf")
    else:
        print("LaTeX compilation failed - no PDF generated!")
        print("First compilation output:")
        print(result1.stderr)
        print("\nSecond compilation output:")
        print(result2.stderr)
        sys.exit(1)

except FileNotFoundError:
    print("Error: pdflatex not found. Please install LaTeX.")
    print("On macOS: brew install --cask mactex")
    print("On Ubuntu: sudo apt-get install texlive-full")
    print("On Windows: Install MiKTeX from https://miktex.org/")
    sys.exit(1)

print("Report generation completed successfully!")
print("Files generated:")
print("   - report.pdf (main report)")
print("   - purity.png, h2o_concentration.png, temperature.png, level.png (plots)")
print("   - report.tex (LaTeX source)")