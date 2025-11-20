"""
Script for correlating Cole-Moore analysis methods using CSV result files

This script reads the CSV results from different Cole-Moore analysis methods
and creates correlation plots comparing derivative analysis with other methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from scipy.stats import pearsonr


def correlate_cole_moore(exp_csv=None, shift_csv=None, derivative_csv=None, invert_shifts=True):
    """
    Create correlation plots between derivative analysis and other Cole-Moore methods.
    
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
    """
    
    # Load data from CSV files
    methods_data = {}
    
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
    
    if 'Derivative Peak' not in methods_data:
        print("Error: Derivative peak data is required for correlation analysis")
        return
    
    # Extract voltage pattern
    voltage_pattern = re.compile(r'(-?\d+) mV')
    
    # Process derivative data (reference for correlation)
    deriv_data = methods_data['Derivative Peak']
    peak_time_col = 'Peak Time (ms)' if 'Peak Time (ms)' in deriv_data.columns else 'Peak/Max Time (ms)'
    
    derivative_dict = {}
    for _, row in deriv_data.iterrows():
        match = voltage_pattern.search(row['Sweep'])
        if match and not pd.isna(row[peak_time_col]):
            voltage = float(match.group(1))
            peak_time = row[peak_time_col]
            if peak_time > 0:  # Only positive values
                derivative_dict[voltage] = peak_time
    
    print(f"Found {len(derivative_dict)} valid derivative peak measurements")
    
    # Create correlation plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
    method_counter = 0
    
    # Process Exponential Fit correlation
    if 'Exponential Fit' in methods_data:
        exp_data = methods_data['Exponential Fit']
        
        exp_x, exp_y = [], []
        for _, row in exp_data.iterrows():
            match = voltage_pattern.search(row['Sweep'])
            if match and not pd.isna(row['Crossing Time (ms)']):
                voltage = float(match.group(1))
                crossing_time = row['Crossing Time (ms)']
                
                if crossing_time > 0 and voltage in derivative_dict:
                    exp_x.append(derivative_dict[voltage])  # Derivative time on x-axis
                    exp_y.append(crossing_time)  # Exponential crossing time on y-axis
        
        if len(exp_x) >= 2:
            exp_x, exp_y = np.array(exp_x), np.array(exp_y)
            r_exp, p_exp = pearsonr(exp_x, exp_y)
            
            # Plot data points
            ax.scatter(exp_x, exp_y, color=colors[method_counter], s=60, alpha=0.7,
                      label=f'Exponential fit (r={r_exp:.3f})')
            
            # Fit and plot trend line
            z = np.polyfit(exp_x, exp_y, 1)
            p_line = np.poly1d(z)
            x_trend = np.linspace(min(exp_x), max(exp_x), 100)
            ax.plot(x_trend, p_line(x_trend), '--', color=colors[method_counter], alpha=0.8)
            
            method_counter += 1
            
            print(f"Exponential Fit vs Derivative correlation: r = {r_exp:.4f}, p = {p_exp:.4e}, n = {len(exp_x)}")
    
    # Process Trace Alignment correlation  
    if 'Trace Alignment' in methods_data:
        shift_data = methods_data['Trace Alignment']
        
        shift_x, shift_y = [], []
        for _, row in shift_data.iterrows():
            match = voltage_pattern.search(row['Sweep'])
            if match and not pd.isna(row['Shift (ms)']):
                voltage = float(match.group(1))
                shift = row['Shift (ms)']
                
                # Process shifts based on invert_shifts parameter
                if invert_shifts:
                    processed_shift = -shift  # Invert shift direction
                else:
                    processed_shift = shift
                
                if voltage in derivative_dict:
                    if invert_shifts and processed_shift >= 0:
                        shift_x.append(derivative_dict[voltage])  # Derivative time on x-axis
                        shift_y.append(processed_shift)  # Processed shift on y-axis
                    elif not invert_shifts:
                        shift_x.append(derivative_dict[voltage])
                        shift_y.append(processed_shift)
        
        if len(shift_x) >= 2:
            shift_x, shift_y = np.array(shift_x), np.array(shift_y)
            r_shift, p_shift = pearsonr(shift_x, shift_y)
            
            # Plot data points
            ax.scatter(shift_x, shift_y, color=colors[method_counter], s=60, alpha=0.7,
                      label=f'Trace Alignment (r={r_shift:.3f})')
            
            # Fit and plot trend line
            z = np.polyfit(shift_x, shift_y, 1)
            p_line = np.poly1d(z)
            x_trend = np.linspace(min(shift_x), max(shift_x), 100)
            ax.plot(x_trend, p_line(x_trend), '--', color=colors[method_counter], alpha=0.8)
            
            print(f"Trace Alignment vs Derivative correlation: r = {r_shift:.4f}, p = {p_shift:.4e}, n = {len(shift_x)}")
    
    # Format plot
    ax.set_xlabel('Derivative Peak Time (ms)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Shift (ms)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nCorrelation analysis complete using derivative peak method as reference")

# Main execution
if __name__ == "__main__":
    print("\n--- Cole-Moore Methods Correlation Analysis ---")
    
    # Specify paths to the CSV files from each method
    exp_file = "cole_moore_exp_fit_results_single.csv"
    shift_file = "cole_moore_shift_results.csv" 
    derivative_file = "cole_moore_derivative_peak_results.csv"
    
    # Check for existing files
    all_files_exist = True
    
    for file_path, method_name in [
        (exp_file, "Exponential Fit"),
        (shift_file, "Trace Alignment"),
        (derivative_file, "Derivative Peak")
    ]:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. {method_name} results will not be included.")
            all_files_exist = False
        else:
            print(f"Found: {file_path}")
    
    if not os.path.exists(derivative_file):
        print("Error: Derivative peak results file is required for correlation analysis")
    else:
        print("\nProceeding with correlation analysis...")
        
        # Run correlation analysis
        correlate_cole_moore(
            exp_csv=exp_file,
            shift_csv=shift_file, 
            derivative_csv=derivative_file,
            invert_shifts=True
        )
    
    print("\n--- Correlation Analysis Complete ---")