import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.stats import pearsonr

def correlate_cole_moore_baselined(exp_csv=None, shift_csv=None, derivative_csv=None, invert_shifts=True):
    """
    Create correlation plots between derivative analysis and other Cole-Moore methods
    using baselined (normalized) shift data.
    
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
        if file_path:
            try:
                methods_data[name] = pd.read_csv(file_path)
                print(f"Loaded {name} data from {file_path}")
            except FileNotFoundError:
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
    derivative_values = []
    
    for _, row in deriv_data.iterrows():
        match = voltage_pattern.search(row['Sweep'])
        if match and not pd.isna(row[peak_time_col]):
            voltage = float(match.group(1))
            peak_time = row[peak_time_col]
            if peak_time > 0:  # Only positive values
                derivative_dict[voltage] = peak_time
                derivative_values.append(peak_time)
    
    # Baseline derivative data (subtract minimum)
    if derivative_values:
        min_derivative = min(derivative_values)
        for voltage in derivative_dict:
            derivative_dict[voltage] -= min_derivative
        print(f"Derivative data baselined (subtracted {min_derivative:.3f} ms)")
    
    print(f"Found {len(derivative_dict)} valid derivative peak measurements")
    
    # Create correlation plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
    method_counter = 0
    
    # Process Exponential Fit correlation
    if 'Exponential Fit' in methods_data:
        exp_data = methods_data['Exponential Fit']
        
        exp_x, exp_y = [], []
        exp_y_raw = []
        
        for _, row in exp_data.iterrows():
            match = voltage_pattern.search(row['Sweep'])
            if match and not pd.isna(row['Crossing Time (ms)']):
                voltage = float(match.group(1))
                crossing_time = row['Crossing Time (ms)']
                
                if crossing_time > 0 and voltage in derivative_dict:
                    exp_x.append(derivative_dict[voltage])  # Baselined derivative time
                    exp_y_raw.append(crossing_time)  # Store raw values for baselining
        
        # Baseline exponential data
        if exp_y_raw:
            min_exp = min(exp_y_raw)
            exp_y = [val - min_exp for val in exp_y_raw]
            print(f"Exponential data baselined (subtracted {min_exp:.3f} ms)")
            
            if len(exp_x) >= 2:
                exp_x, exp_y = np.array(exp_x), np.array(exp_y)
                r_exp, p_exp = pearsonr(exp_x, exp_y)
                
                # Plot data points
                ax.scatter(exp_x, exp_y, color=colors[method_counter], s=80, alpha=0.7,
                          label=f'Exponential fit (r={r_exp:.3f})')
                
                # Fit and plot trend line
                z = np.polyfit(exp_x, exp_y, 1)
                p_line = np.poly1d(z)
                x_trend = np.linspace(min(exp_x), max(exp_x), 100)
                ax.plot(x_trend, p_line(x_trend), '--', color=colors[method_counter], alpha=0.8, linewidth=2)
                
                method_counter += 1
                
                print(f"Exponential Fit vs Derivative correlation: r = {r_exp:.4f}, p = {p_exp:.4e}, n = {len(exp_x)}")
    
    # Process Trace Alignment correlation  
    if 'Trace Alignment' in methods_data:
        shift_data = methods_data['Trace Alignment']
        
        shift_x, shift_y = [], []
        shift_y_raw = []
        
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
                        shift_x.append(derivative_dict[voltage])  # Baselined derivative time
                        shift_y_raw.append(processed_shift)  # Store raw values for baselining
                    elif not invert_shifts:
                        shift_x.append(derivative_dict[voltage])
                        shift_y_raw.append(processed_shift)
        
        # Baseline shift data
        if shift_y_raw:
            min_shift = min(shift_y_raw)
            shift_y = [val - min_shift for val in shift_y_raw]
            print(f"Trace alignment data baselined (subtracted {min_shift:.3f} ms)")
            
            if len(shift_x) >= 2:
                shift_x, shift_y = np.array(shift_x), np.array(shift_y)
                r_shift, p_shift = pearsonr(shift_x, shift_y)
                
                # Plot data points
                ax.scatter(shift_x, shift_y, color=colors[method_counter], s=80, alpha=0.7,
                          label=f'Trace Alignment (r={r_shift:.3f})')
                
                # Fit and plot trend line
                z = np.polyfit(shift_x, shift_y, 1)
                p_line = np.poly1d(z)
                x_trend = np.linspace(min(shift_x), max(shift_x), 100)
                ax.plot(x_trend, p_line(x_trend), '--', color=colors[method_counter], alpha=0.8, linewidth=2)
                
                print(f"Trace Alignment vs Derivative correlation: r = {r_shift:.4f}, p = {p_shift:.4e}, n = {len(shift_x)}")
    
    # Format plot
    ax.set_xlabel('Derivative Peak Time (ms)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Shift (ms)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nCorrelation analysis complete using baselined data")

