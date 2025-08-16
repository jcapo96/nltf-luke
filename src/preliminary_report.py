#!/usr/bin/env python3
"""
Preliminary Report Generator for NLTF-LUKE Data Analysis

This script generates a preliminary report with raw data plots and analysis summaries
before the main analysis is performed. It helps users understand their data quality
and identify potential issues early in the process.
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysisClasses import DatasetManager
from dataFormats import DataFormatManager

# Set up the plotting style
plt.style.use("style.mplstyle")

def create_raw_data_plots(dataset, output_dir, config, dataset_type):
    """Create raw data plots for a dataset with integration window visualization."""
    print(f"🎨 Creating plots for {dataset_type} dataset...")

    # Get integration parameters
    integration_params = config["Parameters"]["H2O"]

    # Get available signals
    signals = []
    if dataset.standard_data.liquid_level is not None:
        signals.append(('liquid_level', 'Liquid Level [mm]'))
    if dataset.standard_data.h2o_concentration is not None:
        signals.append(('h2o_concentration', r'H$_2$O Concentration [ppb]'))
    if dataset.standard_data.temperature is not None:
        signals.append(('temperature', 'Temperature [K]'))
    if dataset.standard_data.purity is not None:
        signals.append(('purity', r'$e^-$ Lifetime [s]'))

    # Create individual plots with progress tracking
    with tqdm(total=len(signals), desc=f"Creating {dataset_type} plots", unit="plot") as pbar:
        for signal_name, signal_label in signals:
            # Get data
            signal_data = getattr(dataset.standard_data, signal_name)
            if signal_data is None:
                continue

            # Create individual plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot raw data
            ax.plot(signal_data.index, signal_data, '-', linewidth=0.8, alpha=0.8, label=f'{dataset_type.capitalize()} {signal_label}')

            # Get find_times results if available
            if dataset.liquid_level:
                level_times = dataset.liquid_level.find_times()
                start_time = level_times.get('start_time')
                end_time = level_times.get('end_time')

                if start_time and end_time:
                    # Mark start and end times
                    ax.axvline(x=start_time, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Run Start')
                    ax.axvline(x=end_time, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Run End')

                    # Calculate integration regions
                    from datetime import timedelta

                    # Initial integration region: integration_time_ini minutes BEFORE t0=0 (run start)
                    ini_start = start_time - timedelta(minutes=integration_params['integration_time_ini'])
                    ini_end = start_time  # t0=0 (run start)

                    # Final integration region: integration_time_end minutes ENDING at offset_end minutes BEFORE end time
                    end_end = end_time - timedelta(minutes=integration_params['offset_end'])  # offset_end minutes before end
                    end_start = end_end - timedelta(minutes=integration_params['integration_time_end'])  # integration_time_end minutes before that

                    # Shade the initial integration region
                    ax.axvspan(ini_start, ini_end, alpha=0.3, color='blue',
                               label=f'Initial Integration: {integration_params["integration_time_ini"]}min before t0')

                    # Shade the final integration region
                    ax.axvspan(end_start, end_end, alpha=0.3, color='orange',
                               label=f'Final Integration: {integration_params["integration_time_end"]}min ending {integration_params["offset_end"]}min before end')

                    # Add vertical lines for integration boundaries
                    ax.axvline(x=ini_start, color='blue', linestyle=':', alpha=0.8, linewidth=1)
                    ax.axvline(x=ini_end, color='blue', linestyle=':', alpha=0.8, linewidth=1)
                    ax.axvline(x=end_start, color='orange', linestyle=':', alpha=0.8, linewidth=1)
                    ax.axvline(x=end_end, color='orange', linestyle=':', alpha=0.8, linewidth=1)

            # Format plot
            ax.set_title(f'{dataset_type.capitalize()} - {signal_label} (Raw Data)')
            ax.set_xlabel('Time')
            ax.set_ylabel(signal_label)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Save individual plot
            plot_filename = os.path.join(output_dir, f"{dataset_type}_{signal_name}_raw.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved {plot_filename}")

            pbar.update(1)

    # Create combined plot
    print(f"🔄 Creating combined plot for {dataset_type}...")
    fig, axes = plt.subplots(len(signals), 1, figsize=(14, 3*len(signals)))
    if len(signals) == 1:
        axes = [axes]

    for i, (signal_name, signal_label) in enumerate(signals):
        ax = axes[i]
        signal_data = getattr(dataset.standard_data, signal_name)

        # Plot raw data
        ax.plot(signal_data.index, signal_data, '-', linewidth=0.8, alpha=0.8, label=f'{dataset_type.capitalize()} {signal_label}')

        # Get find_times results if available
        if dataset.liquid_level:
            level_times = dataset.liquid_level.find_times()
            start_time = level_times.get('start_time')
            end_time = level_times.get('end_time')

            if start_time and end_time:
                # Mark start and end times
                ax.axvline(x=start_time, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Run Start')
                ax.axvline(x=end_time, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Run End')

                # Calculate integration regions
                from datetime import timedelta

                # Initial integration region: integration_time_ini minutes BEFORE t0=0 (run start)
                ini_start = start_time - timedelta(minutes=integration_params['integration_time_ini'])
                ini_end = start_time  # t0=0 (run start)

                # Final integration region: integration_time_end minutes ENDING at offset_end minutes BEFORE end time
                end_end = end_time - timedelta(minutes=integration_params['offset_end'])  # offset_end minutes before end
                end_start = end_end - timedelta(minutes=integration_params['integration_time_end'])  # integration_time_end minutes before that

                # Shade the initial integration region
                ax.axvspan(ini_start, ini_end, alpha=0.3, color='blue')

                # Shade the final integration region
                ax.axvspan(end_start, end_end, alpha=0.3, color='orange')

        # Format subplot
        ax.set_title(f'{signal_label}')
        ax.set_ylabel(signal_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Set common x-label
    axes[-1].set_xlabel('Time')

    # Adjust layout and save
    plt.tight_layout()
    combined_filename = os.path.join(output_dir, f"{dataset_type}_all_signals_raw.png")
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined plot: {combined_filename}")

def generate_preliminary_report(json_file, output_dir="preliminary_plots"):
    """
    Generate a complete preliminary report.

    Args:
        json_file: Path to the JSON configuration file
        output_dir: Directory to save the plots
    """
    print("=" * 60)
    print("PRELIMINARY REPORT GENERATOR")
    print("=" * 60)
    print(f"Configuration file: {json_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    try:
        # Load configuration
        with open(json_file, 'r') as f:
            config = json.load(f)

        # Extract dataset information
        data_path = config["Data"]["Path"]
        data_name = config["Data"]["Name"]

        # Create dataset paths
        dataset_paths = {
            'baseline': f"{data_path}/{data_name}_baseline.xlsx",
            'ullage': f"{data_path}/{data_name}_ullage.xlsx",
            'liquid': f"{data_path}/{data_name}_liquid.xlsx"
        }

        print(f"Dataset paths:")
        for dataset_type, path in dataset_paths.items():
            print(f"  {dataset_type}: {path}")

        # Initialize dataset manager
        print(f"\nInitializing DatasetManager...")
        dataset_manager = DatasetManager(dataset_paths, data_name, json_file)

        # Check which datasets were loaded successfully
        print(f"\nLoaded datasets:")
        for dataset_type, dataset in dataset_manager.datasets.items():
            if dataset is not None:
                print(f"  ✓ {dataset_type}: {dataset.name}")
            else:
                print(f"  ✗ {dataset_type}: Failed to load")

        # Generate raw data plots
        print(f"\nGenerating raw data plots...")
        create_raw_data_plots(dataset_manager, output_dir, config)

        # Generate summary report
        print(f"\nGenerating summary report...")
        generate_summary_report(dataset_manager, config, output_dir)

        print(f"\n" + "=" * 60)
        print("PRELIMINARY REPORT GENERATION COMPLETED!")
        print(f"Check the '{output_dir}' directory for all plots and reports.")
        print("=" * 60)

    except Exception as e:
        print(f"Error generating preliminary report: {e}")
        import traceback
        traceback.print_exc()

def generate_summary_report(dataset_manager, config, output_dir):
    """
    Generate a summary report with dataset information and statistics.

    Args:
        dataset_manager: The DatasetManager instance
        config: The configuration dictionary
        output_dir: Directory to save the report
    """
    report_lines = []
    report_lines.append("PRELIMINARY DATA ANALYSIS REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Configuration: {config.get('Sample', {}).get('Sample Name', 'Unknown')}")
    report_lines.append("")

    # Dataset summary
    report_lines.append("DATASET SUMMARY")
    report_lines.append("-" * 20)

    for dataset_type, dataset in dataset_manager.datasets.items():
        report_lines.append(f"\n{dataset_type.upper()} DATASET:")

        if dataset is None:
            report_lines.append("  Status: Failed to load")
            continue

        report_lines.append(f"  Status: Loaded successfully")
        report_lines.append(f"  Name: {dataset.name}")

        # Get time information
        try:
            level_times = dataset.liquid_level.find_times()
            start_time = level_times['start_time']
            end_time = level_times['end_time']

            if start_time and end_time:
                duration = end_time - start_time
                report_lines.append(f"  Start time: {start_time}")
                report_lines.append(f"  End time: {end_time}")
                report_lines.append(f"  Duration: {duration}")
            else:
                report_lines.append("  Time analysis: Could not determine start/end times")
        except Exception as e:
            report_lines.append(f"  Time analysis: Error - {e}")

        # Signal information
        report_lines.append("  Available signals:")
        signals_info = []

        if hasattr(dataset.standard_data, 'liquid_level') and dataset.standard_data.liquid_level is not None:
            level_data = dataset.standard_data.liquid_level
            signals_info.append(f"    - Liquid Level: {len(level_data)} points, range: {level_data.min():.2f} - {level_data.max():.2f} mm")

        if hasattr(dataset.standard_data, 'h2o_concentration') and dataset.standard_data.h2o_concentration is not None:
            h2o_data = dataset.standard_data.h2o_concentration
            signals_info.append(f"    - H$_2$O Concentration: {len(h2o_data)} points, range: {h2o_data.min():.2f} - {h2o_data.max():.2f} ppb")

        if hasattr(dataset.standard_data, 'temperature') and dataset.standard_data.temperature is not None:
            temp_data = dataset.standard_data.temperature
            signals_info.append(f"    - Temperature: {len(temp_data)} points, range: {temp_data.min():.2f} - {temp_data.max():.2f} °C")

        if hasattr(dataset.standard_data, 'purity') and dataset.standard_data.purity is not None:
            purity_data = dataset.standard_data.purity
            signals_info.append(f"    - Purity: {len(purity_data)} points, range: {purity_data.min():.2f} - {purity_data.max():.2f} arb. units")

        if signals_info:
            report_lines.extend(signals_info)
        else:
            report_lines.append("    - No signals available")

    # Configuration information
    report_lines.append(f"\nCONFIGURATION")
    report_lines.append("-" * 20)
    report_lines.append(f"Converter used: {dataset_manager.preferred_converter}")
    report_lines.append(f"Data path: {config['Data']['Path']}")
    report_lines.append(f"Data name: {config['Data']['Name']}")

    # Integration parameters information
    if "Parameters" in config and "H2O" in config["Parameters"]:
        h2o_params = config["Parameters"]["H2O"]
        report_lines.append(f"\nINTEGRATION PARAMETERS")
        report_lines.append("-" * 20)
        report_lines.append(f"Initial integration time: {h2o_params.get('integration_time_ini', 'N/A')} minutes")
        report_lines.append(f"Final integration time: {h2o_params.get('integration_time_end', 'N/A')} minutes")
        report_lines.append(f"Initial offset: {h2o_params.get('offset_ini', 'N/A')} minutes")
        report_lines.append(f"Final offset: {h2o_params.get('offset_end', 'N/A')} minutes")

        # Calculate and show the actual time windows
        if any(dataset is not None for dataset in dataset_manager.datasets.values()):
            # Get a sample dataset to calculate time windows
            sample_dataset = next((d for d in dataset_manager.datasets.values() if d is not None), None)
            if sample_dataset and hasattr(sample_dataset, 'liquid_level'):
                try:
                    level_times = sample_dataset.liquid_level.find_times()
                    if level_times.get('start_time') and level_times.get('end_time'):
                        start_time = level_times['start_time']
                        end_time = level_times['end_time']

                        from datetime import timedelta

                        # Initial integration window: integration_time_ini minutes BEFORE t0=0 (start_time)
                        ini_start = start_time - timedelta(minutes=h2o_params.get('integration_time_ini', 1))
                        ini_end = start_time  # t0=0 (run start)

                        # Final integration window: integration_time_end minutes ENDING at offset_end minutes BEFORE end_time
                        end_end = end_time - timedelta(minutes=h2o_params.get('offset_end', 0))  # offset_end minutes before end
                        end_start = end_end - timedelta(minutes=h2o_params.get('integration_time_end', 8))  # integration_time_end minutes before that

                        report_lines.append(f"\nCALCULATED INTEGRATION WINDOWS")
                        report_lines.append("-" * 20)
                        report_lines.append(f"Initial integration: {ini_start.strftime('%Y-%m-%d %H:%M:%S')} to {ini_end.strftime('%Y-%m-%d %H:%M:%S')}")
                        report_lines.append(f"Final integration: {end_start.strftime('%Y-%m-%d %H:%M:%S')} to {end_end.strftime('%Y-%m-%d %H:%M:%S')}")

                except Exception as e:
                    report_lines.append(f"Could not calculate integration windows: {e}")

    # Write the report
    report_filename = f"{output_dir}/preliminary_summary.txt"
    with open(report_filename, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Saved summary report: {report_filename}")

def main():
    """Main function to generate preliminary report."""
    print("=" * 60)
    print("PRELIMINARY REPORT GENERATOR")
    print("=" * 60)

    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 src/preliminary_report.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    print(f"Configuration file: {config_file}")

    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{config_file}'.")
        sys.exit(1)

    # Create output directory
    output_dir = "preliminary_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")

    # Extract dataset paths
    data_path = config["Data"]["Path"]
    data_name = config["Data"]["Name"]

    dataset_paths = {
        'baseline': f"{data_path}/{data_name}_baseline.xlsx",
        'ullage': f"{data_path}/{data_name}_ullage.xlsx",
        'liquid': f"{data_path}/{data_name}_liquid.xlsx"
    }

    print(f"📁 Data path: {data_path}")
    print(f"📊 Dataset name: {data_name}")
    print("\nDataset paths:")
    for dataset_type, path in dataset_paths.items():
        print(f"  {dataset_type}: {path}")

    # Initialize dataset manager
    print("\n🚀 Initializing DatasetManager...")
    dataset_manager = DatasetManager(dataset_paths, config["Data"]["Name"], config_file)

    # Load datasets
    print("📊 Loading datasets...")
    with tqdm(total=3, desc="Loading Datasets", unit="dataset") as pbar:
        for dataset_type in ['baseline', 'ullage', 'liquid']:
            if dataset_type in dataset_manager.datasets:
                print(f"  ✓ {dataset_type}: {dataset_manager.datasets[dataset_type].name}")
            pbar.update(1)

    # Generate raw data plots
    print("\n🎨 Generating raw data plots...")
    integration_params = config["Parameters"]["H2O"]
    print(f"Integration parameters: {integration_params}")

    # Create plots for each dataset
    with tqdm(total=3, desc="Processing Datasets", unit="dataset") as pbar:
        for dataset_type in ['baseline', 'ullage', 'liquid']:
            print(f"\n=== Processing {dataset_type.upper()} dataset ===")
            dataset = dataset_manager.datasets[dataset_type]

            if dataset:
                # Get dataset info
                level_times = dataset.liquid_level.find_times() if dataset.liquid_level else {}
                start_time = level_times.get('start_time', 'N/A')
                end_time = level_times.get('end_time', 'N/A')

                if start_time != 'N/A' and end_time != 'N/A':
                    print(f"Start time: {start_time}")
                    print(f"End time: {end_time}")
                else:
                    print("Start/End times: Not available")

                # Show available signals
                available_signals = []
                if dataset.standard_data.liquid_level is not None:
                    available_signals.append('liquid_level')
                if dataset.standard_data.h2o_concentration is not None:
                    available_signals.append('h2o_concentration')
                if dataset.standard_data.temperature is not None:
                    available_signals.append('temperature')
                if dataset.standard_data.purity is not None:
                    available_signals.append('purity')

                print(f"Available signals: {available_signals}")

                # Generate plots
                create_raw_data_plots(dataset, output_dir, config, dataset_type)

            pbar.update(1)

    # Generate summary report
    print("\n📋 Generating summary report...")
    generate_summary_report(dataset_manager, config, output_dir)

    print("\n" + "=" * 60)
    print("PRELIMINARY REPORT GENERATION COMPLETED!")
    print(f"Check the '{output_dir}' directory for all plots and reports.")
    print("=" * 60)

if __name__ == "__main__":
    main()
