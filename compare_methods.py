"""
Cole-Moore Analysis Comparison Module with Baseline Plot

This module provides functions to compare the results from different Cole-Moore
shift analysis methods with both absolute and baseline-normalized plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def compare_cole_moore_methods(exp_csv=None, shift_csv=None, derivative_csv=None, invert_shifts=True):
    """
    Compare Cole-Moore shift results from different analysis methods.
    Creates two plots: absolute values and baseline-normalized values.

    Parameters
    ----------
    exp_csv : str
        Path to exponential fit results CSV
    shift_csv : str
        Path to trace alignment/shift results CSV
    derivative_csv : str
        Path to derivative peak results CSV
    invert_shifts : bool
        If True, invert the shifts from trace alignment (multiply by -1)
        If False, use the shifts as they are without inverting

    Returns
    -------
    None
    """
    methods_data = {}

    # Check if files exist
    for name, file_path in [
        ('Exponential Fit', exp_csv),
        ('Trace Alignment', shift_csv),
        ('Derivative Peak', derivative_csv)
    ]:
        if file_path and os.path.exists(file_path):
            print(f"Loading {name} data from {file_path}")
            methods_data[name] = pd.read_csv(file_path)
        elif file_path:
            print(f"Warning: {file_path} not found, skipping {name} method")

    if not methods_data:
        print("No data files found. Please run the analysis methods first.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5))

    # Regex to extract voltage from sweep names
    voltage_pattern = re.compile(r'(-?\d+) mV')

    markers = ['o', 's', 'D']  # Circle, square, diamond
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green

    # Store processed data for baseline plot
    processed_data = {}

    # Process each method
    for i, (method_name, data) in enumerate(methods_data.items()):
        voltages = []
        values = []
        
        if method_name == 'Exponential Fit':
            # Process exponential fit data
            if 'Sweep' in data.columns and 'Crossing Time (ms)' in data.columns:
                for _, row in data.iterrows():
                    match = voltage_pattern.search(row['Sweep'])
                    if match and not pd.isna(row['Crossing Time (ms)']):
                        voltage = float(match.group(1))
                        crossing_time = row['Crossing Time (ms)']

                        # Only plot positive values
                        if crossing_time > 0:
                            voltages.append(voltage)
                            values.append(crossing_time)

        elif method_name == 'Trace Alignment':
            # Process shift data
            if 'Sweep' in data.columns and 'Shift (ms)' in data.columns:
                for _, row in data.iterrows():
                    match = voltage_pattern.search(row['Sweep'])
                    if match and not pd.isna(row['Shift (ms)']):
                        voltage = float(match.group(1))
                        shift = row['Shift (ms)']

                        # Process shifts based on invert_shifts parameter
                        if invert_shifts:
                            processed_shift = -shift
                        else:
                            processed_shift = shift

                        # Only plot valid values
                        if invert_shifts and processed_shift >= 0:
                            voltages.append(voltage)
                            values.append(processed_shift)
                        elif not invert_shifts:
                            voltages.append(voltage)
                            values.append(processed_shift)

        elif method_name == 'Derivative Peak':
            # Process derivative peak data
            peak_time_col = 'Peak Time (ms)' if 'Peak Time (ms)' in data.columns else 'Peak/Max Time (ms)'
            if 'Sweep' in data.columns and peak_time_col in data.columns:
                for _, row in data.iterrows():
                    match = voltage_pattern.search(row['Sweep'])
                    if match and not pd.isna(row[peak_time_col]):
                        voltage = float(match.group(1))
                        peak_time = row[peak_time_col]

                        # Only plot positive values
                        if peak_time > 0:
                            voltages.append(voltage)
                            values.append(peak_time)

        # Sort data by voltage and plot if we have data
        if voltages and values:
            sorted_data = sorted(zip(voltages, values))
            voltages = [v for v, _ in sorted_data]
            values = [val for _, val in sorted_data]
            
            # Store for baseline plot
            processed_data[method_name] = (voltages, values)
            
            # Plot absolute values
            ax1.plot(voltages, values, '-', color=colors[i],
                    marker=markers[i], markersize=6, linewidth=2,
                    label=f"{method_name}")

    # Now create baseline plot
    for i, (method_name, (voltages, values)) in enumerate(processed_data.items()):
        # Baseline to minimum value
        min_val = min(values)
        baseline_values = [val - min_val for val in values]
        
        # Plot baseline values
        ax2.plot(voltages, baseline_values, '-', color=colors[i],
                marker=markers[i], markersize=6, linewidth=2,
                label=f"{method_name}")

    # Set properties for absolute plot
    ax1.set_xlabel('Holding Voltage (mV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Shift (ms)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)

    # Set properties for baseline plot
    ax2.set_xlabel('Holding Voltage (mV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Shift (ms)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print baseline information
    print("\nBaseline Information:")
    for method_name, (voltages, values) in processed_data.items():
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        print(f"\n{method_name}:")
        print(f"  Original min: {min_val:.4f} ms")
        print(f"  Original max: {max_val:.4f} ms")
        print(f"  Range: {range_val:.4f} ms")

    # Print summary statistics
    print("\nSummary Statistics per Method:")
    for method_name, data in methods_data.items():
        print(f"\n{method_name}:")

        if method_name == 'Exponential Fit':
            if 'Crossing Time (ms)' in data.columns:
                valid_data = data['Crossing Time (ms)'].dropna()
                valid_data = valid_data[valid_data > 0]
                if not valid_data.empty:
                    print(f"  Mean crossing time: {valid_data.mean():.4f} ms")
                    print(f"  Median crossing time: {valid_data.median():.4f} ms")

        elif method_name == 'Trace Alignment':
            if 'Shift (ms)' in data.columns:
                shifts = data['Shift (ms)'].dropna()
                if invert_shifts:
                    processed_shifts = -shifts
                    valid_data = processed_shifts[processed_shifts >= 0]
                    if not valid_data.empty:
                        print(f"  Mean shift: {valid_data.mean():.4f} ms")
                        print(f"  Median shift: {valid_data.median():.4f} ms")
                else:
                    if not shifts.empty:
                        print(f"  Mean shift: {shifts.mean():.4f} ms")
                        print(f"  Median shift: {shifts.median():.4f} ms")

        elif method_name == 'Derivative Peak':
            peak_time_col = 'Peak Time (ms)' if 'Peak Time (ms)' in data.columns else 'Peak/Max Time (ms)'
            if peak_time_col in data.columns:
                valid_data = data[peak_time_col].dropna()
                valid_data = valid_data[valid_data > 0]
                if not valid_data.empty:
                    print(f"  Mean peak time: {valid_data.mean():.4f} ms")
                    print(f"  Median peak time: {valid_data.median():.4f} ms")

# Example usage:
# compare_cole_moore_methods(
#     exp_csv='exponential_results.csv',
#     shift_csv='shift_results.csv', 
#     derivative_csv='derivative_results.csv'
# )