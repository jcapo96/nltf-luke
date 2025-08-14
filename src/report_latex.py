import json
from jinja2 import Template
import subprocess
from analysisClasses import Analysis
from datetime import datetime
import numpy as np
import os
import sys
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
        print("Using PAB_S1_AE_611_AR_REAL_F_CV signal for H2O concentration analysis.")
    elif h2o_parameters["signal"] == "OLD":
        signal_name = "PAB_S1_AE_600_AR_REAL_F_CV_OLD"
        print("Using PAB_S1_AE_600_AR_REAL_F_CV_OLD signal for H2O concentration analysis.")
    else:
        signal_name = h2o_parameters["signal"]
        print(f"Using signal {signal_name} for H2O concentration analysis.")
elif "signal" not in h2o_parameters:
    signal_name = "PAB_S1_AE_611_AR_REAL_F_CV"
    print("Using default signal PAB_S1_AE_611_AR_REAL_F_CV for H2O concentration analysis.")

analysis = Analysis(path=data["Data"]["Path"], name=data["Data"]["Name"])
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig4, ax4 = plt.subplots(figsize=(10, 6))

purity = analysis.purity(show=True, ax=ax1, manual=False)
h2o_concentration = analysis.h2oConcentration(show=True, ax=ax2, manual=h2o_parameters["manual"],
    integration_time_ini=h2o_parameters["integration_time_ini"]*60, integration_time_end=h2o_parameters["integration_time_end"]*60,
    offset_ini=h2o_parameters["offset_ini"]*60, offset_end=h2o_parameters["offset_end"]*60)
temperature = analysis.temperature(show=True, ax=ax3, manual=False)
level = analysis.level(ax=ax4, manual=False)
# Load JSON

fig1.savefig("purity.png", dpi=300)
fig2.savefig("h2o_concentration.png", dpi=300)
fig3.savefig("temperature.png", dpi=300)
fig4.savefig("level.png", dpi=300)

# Fill template
with open("report_template.tex") as f:
    template = Template(f.read())

def format_datetime(dt):
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.strftime("%m/%d/%y %H:%M")

# Replace '%' with '\%' in relevant fields if present
def escape_percent(s):
    return s.replace('%', '\\%') if '%' in s else s

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
        "start_date": format_datetime(h2o_concentration.baseline.start_time),
        "end_date": format_datetime(h2o_concentration.baseline.end_time),
        "initial_concentration": round(h2o_concentration.baseline.h20_ini, 2),
        "final_concentration": round(h2o_concentration.baseline.h20_end, 1),
        "concentration": round(h2o_concentration.baseline.h20_end - h2o_concentration.baseline.h20_ini, 1),
        "initial_concentration_err": round(h2o_concentration.baseline.h20_ini_err, 2),
        "final_concentration_err": round(h2o_concentration.baseline.h20_end_err, 1),
        "concentration_err": round(np.sqrt(h2o_concentration.baseline.h20_end_err**2 + h2o_concentration.baseline.h20_ini_err**2), 1)
    },
    ullage={
        "start_date": format_datetime(h2o_concentration.ullage.start_time),
        "end_date": format_datetime(h2o_concentration.ullage.end_time),
        "initial_concentration": round(h2o_concentration.ullage.h20_ini, 2),
        "final_concentration": round(h2o_concentration.ullage.h20_end, 1),
        "concentration": round(h2o_concentration.ullage.h20_end - h2o_concentration.ullage.h20_ini, 1),
        "initial_concentration_err": round(h2o_concentration.ullage.h20_ini_err, 2),
        "final_concentration_err": round(h2o_concentration.ullage.h20_end_err, 1),
        "concentration_err": round(np.sqrt(h2o_concentration.ullage.h20_end_err**2 + h2o_concentration.ullage.h20_ini_err**2), 1),
        "temperature": round(temperature.ullage.temp_end, 1),
        "temperature_err": round(temperature.ullage.temp_end_err, 1)
    },
    liquid={
        "start_date": format_datetime(h2o_concentration.liquid.start_time),
        "end_date": format_datetime(h2o_concentration.liquid.end_time),
        "initial_concentration": round(h2o_concentration.liquid.h20_ini, 2),
        "final_concentration": round(h2o_concentration.liquid.h20_end, 1),
        "initial_concentration_err": round(h2o_concentration.liquid.h20_ini_err, 2),
        "final_concentration_err": round(h2o_concentration.liquid.h20_end_err, 1),
        "concentration_err": round(np.sqrt(h2o_concentration.liquid.h20_end_err**2 + h2o_concentration.liquid.h20_ini_err**2), 1),
        "concentration": round(h2o_concentration.liquid.h20_end - h2o_concentration.liquid.h20_ini, 1),
        "temperature": round(temperature.liquid.temp_end, 1),
        "temperature_err": round(temperature.liquid.temp_end_err, 1)
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
            "integration_time_ini": round(h2o_parameters["integration_time_ini"]*60, 0),
            "integration_time_end": h2o_parameters["integration_time_end"],
            "offset_ini": round(h2o_parameters["offset_ini"]*60, 0),
            "offset_end": h2o_parameters["offset_end"]
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
