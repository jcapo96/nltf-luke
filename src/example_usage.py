#!/usr/bin/env python3
"""
Example usage of the restructured NLTF-LUKE framework.

This script demonstrates the new modular structure and how to use
the specialized processor and analysis classes.
"""

import matplotlib.pyplot as plt
from dataClasses import Dataset, LiquidLevelProcessor
from analysisClasses import Analysis, PurityAnalysis


def example_single_dataset():
    """Example of using a single dataset with the new structure."""
    print("=== Single Dataset Example ===")
    
    # Create dataset (replace with actual path)
    dataset = Dataset("path/to/your/data.xlsx")
    
    # Load data
    dataset.load()
    dataset.assign_datetime()
    
    # Use specialized processors directly
    if dataset.liquid_level is not None:
        # Find key time points
        max_level, max_time, min_level, min_time, start_time, end_time = \
            dataset.liquid_level.find_times(scan_time=10, threshold=0.995)
        
        print(f"Liquid level analysis:")
        print(f"  Max level: {max_level:.2f} at {max_time}")
        print(f"  Min level: {min_level:.2f} at {min_time}")
        print(f"  Analysis period: {start_time} to {end_time}")
        
        # Plot liquid level
        fig, ax = plt.subplots(figsize=(10, 6))
        dataset.liquid_level.plot_level(start_time, end_time, ax=ax, color='blue')
        plt.title("Liquid Level Analysis")
        plt.show()
    
    # Use other processors
    if dataset.purity is not None:
        purity_data = dataset.purity.calculate_purity(start_time, end_time)
        print(f"Purity analysis completed with {len(purity_data['data'])} data points")


def example_multi_dataset_analysis():
    """Example of using the multi-dataset analysis framework."""
    print("\n=== Multi-Dataset Analysis Example ===")
    
    # Create analysis object (replace with actual path)
    analysis = Analysis("path/to/your/data/directory", "EXPERIMENT_NAME")
    
    # Analyze purity across all datasets
    print("Analyzing purity...")
    analysis.purity(fit_legend=True)
    
    # Analyze temperature
    print("\nAnalyzing temperature...")
    analysis.temperature()
    
    # Analyze H2O concentration
    print("\nAnalyzing H2O concentration...")
    analysis.h2oConcentration(
        integration_time_ini=60,
        integration_time_end=180,
        offset_ini=60,
        offset_end=480
    )
    
    # Analyze liquid level
    print("\nAnalyzing liquid level...")
    analysis.level(fit_legend=True)
    
    # Get all results
    results = analysis.get_analysis_results()
    print(f"\nAnalysis completed. Results available for {len(results)} analysis types.")


def example_custom_analysis():
    """Example of creating custom analysis using the base classes."""
    print("\n=== Custom Analysis Example ===")
    
    # Create dataset manager
    from analysisClasses import DatasetManager
    dataset_manager = DatasetManager("path/to/your/data/directory", "EXPERIMENT_NAME")
    
    # Prepare datasets
    prepared_data = dataset_manager.prepare_datasets(manual=False)
    
    # Custom analysis logic
    for dataset_type, data in prepared_data.items():
        dataset = data['dataset']
        start_time, end_time = data['times'][4], data['times'][5]
        
        print(f"\n{dataset_type.capitalize()} dataset:")
        print(f"  Analysis period: {start_time} to {end_time}")
        
        # Access specific processors
        if dataset.temperature is not None:
            temp_data = dataset.temperature.calculate_temperature(start_time, end_time)
            print(f"  Temperature: {temp_data['initial']:.1f} K → {temp_data['final']:.1f} K")
        
        if dataset.h2o_concentration is not None:
            h2o_data = dataset.h2o_concentration.calculate_concentration(start_time, end_time)
            print(f"  H2O: {h2o_data['initial']:.1f} ppb → {h2o_data['final']:.1f} ppb")


def main():
    """Main function demonstrating the framework usage."""
    print("NLTF-LUKE Framework Example Usage")
    print("=" * 50)
    
    # Note: These examples require actual data files
    # Uncomment and modify the paths to run with your data
    
    # example_single_dataset()
    # example_multi_dataset_analysis()
    # example_custom_analysis()
    
    print("\nExamples completed. Modify the paths in the script to run with actual data.")
    print("The new structure provides:")
    print("- Better separation of concerns")
    print("- More modular and maintainable code")
    print("- Specialized processors for different data types")
    print("- Improved error handling and validation")
    print("- Cleaner analysis workflow")


if __name__ == "__main__":
    main()
