#!/usr/bin/env python3
"""
Preliminary Report Generator

This script generates a preliminary report showing raw data for all datasets
and signals before any processing. It serves as a starting point for further analysis.
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysisClasses import DatasetManager
from dataFormats import DataFormatManager

def create_raw_data_plots(dataset_manager, output_dir="preliminary_plots", config=None):
    """
    Create raw data plots for all datasets and signals.

    Args:
        dataset_manager: The DatasetManager instance
        output_dir: Directory to save the plots
        config: Configuration dictionary containing integration parameters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use("style.mplstyle")

    # Extract integration parameters from config
    integration_params = {}
    if config and "Parameters" in config and "H2O" in config["Parameters"]:
        h2o_params = config["Parameters"]["H2O"]
        integration_params = {
            'integration_time_ini': h2o_params.get('integration_time_ini', 1),  # Already in minutes
            'integration_time_end': h2o_params.get('integration_time_end', 8),  # Already in minutes
            'offset_ini': h2o_params.get('offset_ini', 1),  # Already in minutes
            'offset_end': h2o_params.get('offset_end', 0)   # Already in minutes
        }
        print(f"Integration parameters: {integration_params}")

    # Process each dataset
    for dataset_type, dataset in dataset_manager.datasets.items():
        print(f"\n=== Processing {dataset_type.upper()} dataset ===")

        if dataset is None or not hasattr(dataset, 'standard_data'):
            print(f"Warning: {dataset_type} dataset is not properly loaded")
            continue

        # Get the start and end times from liquid level processor
        try:
            level_times = dataset.liquid_level.find_times()
            start_time = level_times['start_time']
            end_time = level_times['end_time']
            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")
        except Exception as e:
            print(f"Warning: Could not determine start/end times for {dataset_type}: {e}")
            start_time = None
            end_time = None

        # Create plots for each available signal
        signals_to_plot = []

        # Check which signals are available
        if hasattr(dataset.standard_data, 'liquid_level') and dataset.standard_data.liquid_level is not None:
            signals_to_plot.append(('liquid_level', 'Liquid Level (mm)', 'Level'))

        if hasattr(dataset.standard_data, 'h2o_concentration') and dataset.standard_data.h2o_concentration is not None:
            signals_to_plot.append(('h2o_concentration', r'H$_2$O Concentration (ppb)', 'H2O'))

        if hasattr(dataset.standard_data, 'temperature') and dataset.standard_data.temperature is not None:
            signals_to_plot.append(('temperature', 'Temperature (°C)', 'Temperature'))

        if hasattr(dataset.standard_data, 'purity') and dataset.standard_data.purity is not None:
            signals_to_plot.append(('purity', 'Purity (arb. units)', 'Purity'))

        print(f"Available signals: {[s[0] for s in signals_to_plot]}")

        # Create individual plots for each signal
        for signal_name, ylabel, title_suffix in signals_to_plot:
            try:
                # Get the signal data
                signal_data = getattr(dataset.standard_data, signal_name)
                if signal_data is None or signal_data.empty:
                    print(f"Warning: {signal_name} data is empty for {dataset_type}")
                    continue

                # Create the plot
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot the raw data
                ax.plot(signal_data.index, signal_data.values,
                       linewidth=1, alpha=0.8, label=f'{dataset_type.capitalize()} {title_suffix}')

                # Add start and end time markers if available
                if start_time is not None and end_time is not None:
                    # Convert to datetime if needed
                    if hasattr(start_time, 'to_pydatetime'):
                        start_dt = start_time.to_pydatetime()
                    else:
                        start_dt = start_time

                    if hasattr(end_time, 'to_pydatetime'):
                        end_dt = end_time.to_pydatetime()
                    else:
                        end_dt = end_time

                    # Add vertical lines for start and end times
                    ax.axvline(x=start_dt, color='green', linestyle='--', alpha=0.7,
                              label=f'Start: {start_dt.strftime("%H:%M:%S")}')
                    ax.axvline(x=end_dt, color='red', linestyle='--', alpha=0.7,
                              label=f'End: {end_dt.strftime("%H:%M:%S")}')

                    # Add integration time and offset regions if parameters are available
                    if integration_params and signal_name in ['h2o_concentration', 'temperature', 'purity']:
                        # Calculate integration regions
                        from datetime import timedelta

                        # Initial integration region: integration_time_ini minutes BEFORE t0=0 (run start)
                        ini_start = start_dt - timedelta(minutes=integration_params['integration_time_ini'])
                        ini_end = start_dt  # t0=0 (run start)

                        # Final integration region: integration_time_end minutes ENDING at offset_end minutes BEFORE end time
                        end_end = end_dt - timedelta(minutes=integration_params['offset_end'])  # offset_end minutes before end
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

                # Customize the plot
                ax.set_xlabel('Time')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{dataset_type.capitalize()} Dataset - Raw {title_suffix} Data')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                # Adjust layout and save
                plt.tight_layout()
                plot_filename = f"{output_dir}/{dataset_type}_{signal_name}_raw.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"✓ Saved {plot_filename}")

            except Exception as e:
                print(f"Error plotting {signal_name} for {dataset_type}: {e}")

        # Create a combined plot showing all signals for this dataset
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{dataset_type.capitalize()} Dataset - All Raw Signals', fontsize=16)

            # Flatten axes for easier iteration
            axes_flat = axes.flatten()

            for i, (signal_name, ylabel, title_suffix) in enumerate(signals_to_plot):
                if i >= len(axes_flat):
                    break

                ax = axes_flat[i]
                signal_data = getattr(dataset.standard_data, signal_name)

                if signal_data is not None and not signal_data.empty:
                    ax.plot(signal_data.index, signal_data.values,
                           linewidth=1, alpha=0.8)

                    # Add start and end time markers if available
                    if start_time is not None and end_time is not None:
                        if hasattr(start_time, 'to_pydatetime'):
                            start_dt = start_time.to_pydatetime()
                        else:
                            start_dt = start_time

                        if hasattr(end_time, 'to_pydatetime'):
                            end_dt = end_time.to_pydatetime()
                        else:
                            end_dt = end_time

                        ax.axvline(x=start_dt, color='green', linestyle='--', alpha=0.7)
                        ax.axvline(x=end_dt, color='red', linestyle='--', alpha=0.7)

                        # Add integration time and offset regions if parameters are available
                        if integration_params and signal_name in ['h2o_concentration', 'temperature', 'purity']:
                            # Calculate integration regions
                            from datetime import timedelta

                            # Initial integration region: integration_time_ini minutes BEFORE t0=0 (run start)
                            ini_start = start_dt - timedelta(minutes=integration_params['integration_time_ini'])
                            ini_end = start_dt  # t0=0 (run start)

                            # Final integration region: integration_time_end minutes ENDING at offset_end minutes BEFORE end time
                            end_end = end_dt - timedelta(minutes=integration_params['offset_end'])  # offset_end minutes before end
                            end_start = end_end - timedelta(minutes=integration_params['integration_time_end'])  # integration_time_end minutes before that

                            # Shade the initial integration region
                            ax.axvspan(ini_start, ini_end, alpha=0.3, color='blue')

                            # Shade the final integration region
                            ax.axvspan(end_start, end_end, alpha=0.3, color='orange')

                            # Add vertical lines for integration boundaries
                            ax.axvline(x=ini_start, color='blue', linestyle=':', alpha=0.8, linewidth=1)
                            ax.axvline(x=ini_end, color='blue', linestyle=':', alpha=0.8, linewidth=1)
                            ax.axvline(x=end_start, color='orange', linestyle=':', alpha=0.8, linewidth=1)
                            ax.axvline(x=end_end, color='orange', linestyle=':', alpha=0.8, linewidth=1)

                    ax.set_xlabel('Time')
                    ax.set_ylabel(ylabel)
                    ax.set_title(f'{title_suffix}')
                    ax.grid(True, alpha=0.3)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Hide unused subplots
            for i in range(len(signals_to_plot), len(axes_flat)):
                axes_flat[i].set_visible(False)

            plt.tight_layout()
            combined_filename = f"{output_dir}/{dataset_type}_all_signals_raw.png"
            plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ Saved combined plot: {combined_filename}")

        except Exception as e:
            print(f"Error creating combined plot for {dataset_type}: {e}")

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <input.json>")
        sys.exit(1)

    json_file = sys.argv[1]
    generate_preliminary_report(json_file)
