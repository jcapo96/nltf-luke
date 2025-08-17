#!/usr/bin/env python3
"""
Preliminary report generator for NLTF-LUKE data analysis.
Generates raw data plots and summary for initial data inspection.
"""

import os
import sys
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
from tqdm import tqdm

# Suppress OpenPyXL warnings about default styles
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DatasetManager

plt.style.use("style.mplstyle")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/preliminary_report.py <json_file>")
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

    output_dir = "preliminary_plots"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PRELIMINARY REPORT GENERATOR")
    print("=" * 60)
    print(f"Configuration file: {json_file}")
    print(f"Output directory: {output_dir}")
    print(f"Data path: {data_path}")
    print(f"Dataset name: {data_name}")
    print()
    print("Dataset paths:")
    for dataset_type, path in dataset_paths.items():
        print(f"  {dataset_type}: {path}")
    print()

    print("Initializing DatasetManager...")
    dataset_manager = DatasetManager(dataset_paths, data_name, json_file)

    print("Loading datasets...")
    with tqdm(total=3, desc="Loading Datasets", unit="dataset") as pbar:
        for dataset_type in ['baseline', 'ullage', 'liquid']:
            dataset_manager.load_dataset(dataset_type)
            print(f"  {dataset_type}: {dataset_type}")
            pbar.update(1)

    print("Generating raw data plots...")

    h2o_parameters = data["Parameters"]["H2O"]
    integration_params = {
        'manual': h2o_parameters.get('manual', False),
        'integration_time_ini': h2o_parameters.get('integration_time_ini', 60),
        'integration_time_end': h2o_parameters.get('integration_time_end', 480),
        'offset_ini': h2o_parameters.get('offset_ini', 60),
        'offset_end': h2o_parameters.get('offset_end', 0)
    }

    print(f"Integration parameters: {integration_params}")

    with tqdm(total=3, desc="Processing Datasets", unit="dataset") as pbar:
        for dataset_type in ['baseline', 'ullage', 'liquid']:
            print(f"=== Processing {dataset_type.upper()} dataset ===")

            dataset = dataset_manager.datasets[dataset_type]
            if dataset is None:
                print(f"  Warning: {dataset_type} dataset not loaded")
                pbar.update(1)
                continue

            if dataset.liquid_level is not None:
                level_times = dataset.liquid_level.find_times()
                start_time = level_times['start_time']
                end_time = level_times['end_time']
                print(f"Start time: {start_time}")
                print(f"End time: {end_time}")
            else:
                start_time = None
                end_time = None
                print("No liquid level data available")

            available_signals = []
            if dataset.liquid_level is not None:
                available_signals.append('liquid_level')
            if dataset.h2o_concentration is not None:
                available_signals.append('h2o_concentration')
            if dataset.temperature is not None:
                available_signals.append('temperature')
            if dataset.purity is not None:
                available_signals.append('purity')

            print(f"Available signals: {available_signals}")

            print(f"Creating plots for {dataset_type} dataset...")
            create_raw_data_plots(dataset, output_dir, data, dataset_type, start_time, end_time, integration_params)
            pbar.update(1)

    print("Generating summary report...")
    generate_summary_report(data, dataset_manager, output_dir)

    print("=" * 60)
    print("PRELIMINARY REPORT GENERATION COMPLETED!")
    print(f"Check the '{output_dir}' directory for all plots and reports.")
    print("=" * 60)

def create_raw_data_plots(dataset, output_dir, config, dataset_type, start_time, end_time, integration_params):
    with tqdm(total=4, desc=f"Creating {dataset_type} plots", unit="plot") as pbar:
        if dataset.liquid_level is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            dataset.liquid_level.plot_level(start_time, end_time, ax=ax, dataset_name=dataset_type.capitalize())
            plot_filename = os.path.join(output_dir, f"{dataset_type}_liquid_level_raw.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plot_filename}")
            pbar.update(1)

        if dataset.h2o_concentration is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            dataset.h2o_concentration.plot_h2o_concentration(start_time, end_time, ax=ax, dataset_name=dataset_type.capitalize())
            plot_filename = os.path.join(output_dir, f"{dataset_type}_h2o_concentration_raw.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plot_filename}")
            pbar.update(1)

        if dataset.temperature is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            dataset.temperature.plot_temperature(start_time, end_time, ax=ax, dataset_name=dataset_type.capitalize())
            plot_filename = os.path.join(output_dir, f"{dataset_type}_temperature_raw.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plot_filename}")
            pbar.update(1)

        if dataset.purity is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            dataset.purity.plot_purity(start_time, end_time, {}, ax=ax, dataset_name=dataset_type.capitalize())
            plot_filename = os.path.join(output_dir, f"{dataset_type}_purity_raw.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plot_filename}")
            pbar.update(1)

    print(f"Creating combined plot for {dataset_type}...")
    create_combined_plot(dataset, output_dir, dataset_type, start_time, end_time)

def create_combined_plot(dataset, output_dir, dataset_type, start_time, end_time):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{dataset_type.capitalize()} Dataset - All Signals', fontsize=16)

    if dataset.liquid_level is not None:
        dataset.liquid_level.plot_level(start_time, end_time, ax=axes[0, 0], dataset_name=dataset_type.capitalize())
        axes[0, 0].set_title('Liquid Level')

    if dataset.h2o_concentration is not None:
        dataset.h2o_concentration.plot_h2o_concentration(start_time, end_time, ax=axes[0, 1], dataset_name=dataset_type.capitalize())
        axes[0, 1].set_title('H₂O Concentration')

    if dataset.temperature is not None:
        dataset.temperature.plot_temperature(start_time, end_time, ax=axes[1, 0], dataset_name=dataset_type.capitalize())
        axes[1, 0].set_title('Temperature')

    if dataset.purity is not None:
        dataset.purity.plot_purity(start_time, end_time, {}, ax=axes[1, 1], dataset_name=dataset_type.capitalize())
        axes[1, 1].set_title('Purity')

    plt.tight_layout()
    combined_filename = os.path.join(output_dir, f"{dataset_type}_all_signals_raw.png")
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {combined_filename}")

def generate_summary_report(data, dataset_manager, output_dir):
    summary_file = os.path.join(output_dir, "preliminary_summary.txt")

    with open(summary_file, 'w') as f:
        f.write("PRELIMINARY DATA ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {data['Sample']['Sample Name']}\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 20 + "\n\n")

        for dataset_type in ['baseline', 'ullage', 'liquid']:
            dataset = dataset_manager.datasets[dataset_type]
            if dataset is None:
                f.write(f"{dataset_type.upper()} DATASET:\n")
                f.write(f"  Status: Not loaded\n\n")
                continue

            f.write(f"{dataset_type.upper()} DATASET:\n")
            f.write(f"  Status: Loaded successfully\n")
            f.write(f"  Name: {dataset_type}\n")

            if dataset.liquid_level is not None:
                level_times = dataset.liquid_level.find_times()
                start_time = level_times['start_time']
                end_time = level_times['end_time']
                if start_time and end_time:
                    duration = end_time - start_time
                    f.write(f"  Start time: {start_time}\n")
                    f.write(f"  End time: {end_time}\n")
                    f.write(f"  Duration: {duration}\n")

            available_signals = []
            signal_info = []

            if dataset.liquid_level is not None:
                available_signals.append('liquid_level')
                level_data = dataset.standard_data.liquid_level
                if level_data is not None:
                    signal_info.append(f"Liquid Level: {len(level_data)} points, range: {level_data.min():.2f} - {level_data.max():.2f} mm")

            if dataset.h2o_concentration is not None:
                available_signals.append('h2o_concentration')
                h2o_data = dataset.standard_data.h2o_concentration
                if h2o_data is not None:
                    signal_info.append(f"H$_2$O Concentration: {len(h2o_data)} points, range: {h2o_data.min():.2f} - {h2o_data.max():.2f} ppb")

            if dataset.temperature is not None:
                available_signals.append('temperature')
                temp_data = dataset.standard_data.temperature
                if temp_data is not None:
                    signal_info.append(f"Temperature: {len(temp_data)} points, range: {temp_data.min():.2f} - {temp_data.max():.2f} °C")

            if dataset.purity is not None:
                available_signals.append('purity')
                purity_data = dataset.standard_data.purity
                if purity_data is not None:
                    signal_info.append(f"Purity: {len(purity_data)} points, range: {purity_data.min():.2f} - {purity_data.max():.2f} arb. units")

            f.write(f"  Available signals: {available_signals}\n")
            for info in signal_info:
                f.write(f"    - {info}\n")
            f.write("\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Converter used: {data['Data'].get('Converter', 'Default')}\n")
        f.write(f"Data path: {data['Data']['Path']}\n")
        f.write(f"Data name: {data['Data']['Name']}\n\n")

        f.write("INTEGRATION PARAMETERS\n")
        f.write("-" * 20 + "\n")
        h2o_params = data["Parameters"]["H2O"]
        f.write(f"Initial integration time: {h2o_params.get('integration_time_ini', 60)} minutes\n")
        f.write(f"Final integration time: {h2o_params.get('integration_time_end', 480)} minutes\n")
        f.write(f"Initial offset: {h2o_params.get('offset_ini', 60)} minutes\n")
        f.write(f"Final offset: {h2o_params.get('offset_end', 0)} minutes\n\n")

        f.write("CALCULATED INTEGRATION WINDOWS\n")
        f.write("-" * 20 + "\n")
        baseline_dataset = dataset_manager.datasets['baseline']
        if baseline_dataset and baseline_dataset.liquid_level is not None:
            level_times = baseline_dataset.liquid_level.find_times()
            if level_times['start_time'] and level_times['end_time']:
                start_time = level_times['start_time']
                ini_start = start_time - timedelta(minutes=h2o_params.get('integration_time_ini', 60))
                ini_end = start_time
                end_end = level_times['end_time'] - timedelta(minutes=h2o_params.get('offset_end', 0))
                end_start = end_end - timedelta(minutes=h2o_params.get('integration_time_end', 480))

                f.write(f"Initial integration: {ini_start} to {ini_end}\n")
                f.write(f"Final integration: {end_start} to {end_end}\n")

    print(f"Saved summary report: {summary_file}")

if __name__ == "__main__":
    main()
