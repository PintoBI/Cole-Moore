"""
Trace Alignment Module for Cole-Moore Analysis

This module provides robust functions to align electrophysiology traces by finding the
optimal time shifts using multiple alignment methods to handle traces with complex kinetics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from IPython.display import display


def align_traces(data, time_col='Time (ms)', reference_sweep=None, start_time=None,
                rising_phase_only=True, alignment_threshold=0.5, 
                alignment_method='multi_threshold', threshold_points=5,
                exclude_outliers=True, max_shift=10.0, auto_detect_polarity=True):
    """
    Align traces using various robust methods to handle traces with different kinetics.
    Modified to better handle inactivation and negative currents.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing time and current traces
    time_col : str
        Name of the time column
    reference_sweep : str or None
        Sweep to use as reference (None uses most typical sweep)
    start_time : float or None
        Minimum time to consider (None uses all data)
    rising_phase_only : bool
        If True, only use the rising phase for alignment
    alignment_threshold : float
        Fraction of maximum amplitude for point of alignment (0.0-1.0)
    alignment_method : str
        'single_threshold': Align at a single threshold point
        'multi_threshold': Align using multiple thresholds (more robust)
        'curve_fit': Align by minimizing overall difference between curves
    threshold_points : int
        Number of threshold points to use for 'multi_threshold' method
    exclude_outliers : bool
        If True, attempt to exclude outlier traces with unusual kinetics
    max_shift : float
        Maximum allowable shift in ms
    auto_detect_polarity : bool
        If True, automatically detect positive vs negative currents

    Returns
    -------
    dict
        Dictionary of time shifts for each sweep
    """
    time = data[time_col].values
    sweep_columns = [col for col in data.columns if col != time_col]

    # Create mask for time values if start_time is specified
    if start_time is not None:
        time_mask = time >= start_time
    else:
        time_mask = np.ones_like(time, dtype=bool)

    print(f"Processing {len(sweep_columns)} sweep columns for alignment")
    print(f"Alignment method: {alignment_method}")
    
    if rising_phase_only:
        print("Using rising phase only for alignment (before inactivation)")

    # Filter out any empty or invalid columns
    valid_sweep_columns = []
    for col in sweep_columns:
        valid_indices = ~np.isnan(data[col].values) & time_mask
        if np.sum(valid_indices) >= 5:  # Need at least 5 valid points
            valid_sweep_columns.append(col)

    if not valid_sweep_columns:
        print("No valid sweep columns found for alignment.")
        return {}
        
    # Identify trace types and possible outliers
    trace_stats = {}
    for sweep in valid_sweep_columns:
        current = data[sweep].values
        valid_indices = ~np.isnan(current) & time_mask
        valid_time = time[valid_indices]
        valid_current = current[valid_indices]
        
        # Detect if this is a negative current using the later part of the trace
        # where currents tend to stabilize
        is_negative = False
        if auto_detect_polarity:
            stable_segment = valid_current[-int(len(valid_current)/5):]  # Use the last 20%
            is_negative = np.mean(stable_segment) < 0
            
        # For analysis, if negative, invert the current but keep original for display
        analysis_current = -valid_current if is_negative else valid_current
        
        # Find the activation (rising) phase by looking for the peak
        peak_idx = np.argmax(analysis_current)
        peak_time = valid_time[peak_idx] if peak_idx < len(valid_time) else np.nan
        
        # Get only the rising phase for analysis
        rising_indices = np.arange(peak_idx + 1)  # Include the peak point
        rising_time = valid_time[:peak_idx+1] if peak_idx > 0 else valid_time
        rising_current = analysis_current[:peak_idx+1] if peak_idx > 0 else analysis_current
        
        # Calculate time to reach 20%, 50%, and 80% of peak
        if peak_idx > 5:
            peak_value = analysis_current[peak_idx]
            base_value = np.min(analysis_current[:peak_idx])
            amplitude = peak_value - base_value
            
            threshold_times = {}
            for pct in [0.2, 0.5, 0.8]:
                threshold = base_value + amplitude * pct
                idx_before = np.where(rising_current <= threshold)[0]
                idx_after = np.where(rising_current >= threshold)[0]
                
                if len(idx_before) > 0 and len(idx_after) > 0:
                    # Find the crossing point
                    idx1 = idx_before[-1]
                    idx2 = idx_after[0]
                    
                    if idx1 == idx2:
                        # Exact match
                        threshold_times[pct] = rising_time[idx1]
                    else:
                        # Linear interpolation
                        t1, t2 = rising_time[idx1], rising_time[idx2]
                        v1, v2 = rising_current[idx1], rising_current[idx2]
                        
                        if v2 - v1 > 1e-10:  # Avoid division by zero
                            t_interp = t1 + (threshold - v1) * (t2 - t1) / (v2 - v1)
                            threshold_times[pct] = t_interp
                        else:
                            threshold_times[pct] = (t1 + t2) / 2
                else:
                    threshold_times[pct] = np.nan
            
            # Calculate activation speed metrics
            if 0.2 in threshold_times and 0.8 in threshold_times and not np.isnan(threshold_times[0.2]) and not np.isnan(threshold_times[0.8]):
                activation_time = threshold_times[0.8] - threshold_times[0.2]
            else:
                activation_time = np.nan
        else:
            threshold_times = {0.2: np.nan, 0.5: np.nan, 0.8: np.nan}
            activation_time = np.nan
            
        trace_stats[sweep] = {
            'peak_time': peak_time,
            'peak_idx': peak_idx,
            'threshold_times': threshold_times,
            'activation_time': activation_time,
            'is_negative': is_negative,
            'rising_indices': rising_indices if peak_idx > 0 else None
        }
    
    # Exclude outliers if requested
    if exclude_outliers:
        # Get activation times
        activation_times = [stats['activation_time'] for stats in trace_stats.values() 
                          if not np.isnan(stats['activation_time'])]
        
        if activation_times:
            # Calculate median and standard deviation
            median_time = np.median(activation_times)
            std_time = np.std(activation_times)
            
            # Define outlier threshold (3 standard deviations from median)
            outlier_threshold = max(0.5, 3 * std_time)  # At least 0.5 ms difference
            
            # Mark outliers
            for sweep, stats in trace_stats.items():
                if not np.isnan(stats['activation_time']):
                    if abs(stats['activation_time'] - median_time) > outlier_threshold:
                        trace_stats[sweep]['is_outlier'] = True
                        print(f"Marked {sweep} as outlier (activation time: {stats['activation_time']:.3f} ms, median: {median_time:.3f} ms)")
                    else:
                        trace_stats[sweep]['is_outlier'] = False
                else:
                    trace_stats[sweep]['is_outlier'] = True  # No activation time available
        
    # Select reference sweep
    if reference_sweep is None:
        # Use the most typical sweep (closest to median activation time)
        non_outlier_sweeps = [sweep for sweep, stats in trace_stats.items() 
                            if not stats.get('is_outlier', False)]
        
        if non_outlier_sweeps:
            # Get activation times for non-outliers
            act_times = [trace_stats[sweep]['activation_time'] for sweep in non_outlier_sweeps 
                       if not np.isnan(trace_stats[sweep]['activation_time'])]
            
            if act_times:
                median_time = np.median(act_times)
                
                # Find sweep closest to median
                best_sweep = None
                min_diff = float('inf')
                
                for sweep in non_outlier_sweeps:
                    act_time = trace_stats[sweep]['activation_time']
                    if not np.isnan(act_time):
                        diff = abs(act_time - median_time)
                        if diff < min_diff:
                            min_diff = diff
                            best_sweep = sweep
                
                if best_sweep:
                    reference_sweep = best_sweep
                    print(f"Selected {reference_sweep} as reference (most typical activation kinetics)")
                else:
                    reference_sweep = non_outlier_sweeps[0]
                    print(f"Using {reference_sweep} as reference (first non-outlier)")
            else:
                reference_sweep = non_outlier_sweeps[0]
                print(f"Using {reference_sweep} as reference (first non-outlier)")
        else:
            # Fall back to last sweep
            reference_sweep = valid_sweep_columns[-1]
            print(f"No suitable reference found. Using last sweep {reference_sweep} as reference")
    elif reference_sweep not in valid_sweep_columns:
        print(f"Specified reference sweep {reference_sweep} not valid. Using automatic selection.")
        reference_sweep = None
        return align_traces(data, time_col, reference_sweep, start_time, rising_phase_only, 
                         alignment_threshold, alignment_method, threshold_points, exclude_outliers, max_shift)
    else:
        print(f"Using specified sweep {reference_sweep} as reference")

    # Initialize results dictionary
    shifts = {reference_sweep: 0.0}  # Reference has zero shift

    # Get reference sweep data
    ref_current = data[reference_sweep].values
    ref_valid = ~np.isnan(ref_current) & time_mask
    ref_time = time[ref_valid]
    ref_current_valid = ref_current[ref_valid]
    
    # Detect if reference is negative and handle accordingly
    ref_stats = trace_stats[reference_sweep]
    is_ref_negative = ref_stats['is_negative']
    ref_analysis_current = -ref_current_valid if is_ref_negative else ref_current_valid
    
    # Find peak in reference
    ref_peak_idx = ref_stats['peak_idx']
    ref_peak_time = ref_stats['peak_time']
    
    # If using rising phase only, restrict the reference data
    if rising_phase_only and ref_peak_idx > 0:
        ref_time = ref_time[:ref_peak_idx+1]  # Include peak
        ref_current_valid = ref_current_valid[:ref_peak_idx+1]
        ref_analysis_current = ref_analysis_current[:ref_peak_idx+1]
        print(f"Reference rising phase: 0 to {ref_peak_time:.3f} ms ({ref_peak_idx+1} points)")
    
    # Create reference data points for alignment
    if alignment_method == 'single_threshold':
        # Find the alignment point at threshold% of maximum
        ref_amplitude = np.max(ref_analysis_current)
        ref_threshold_value = ref_amplitude * alignment_threshold
        
        # Find time at which reference crosses the threshold
        ref_threshold_idx = np.argmin(np.abs(ref_analysis_current - ref_threshold_value))
        ref_threshold_time = ref_time[ref_threshold_idx]
        
        print(f"Reference alignment point at {ref_threshold_time:.4f} ms ({alignment_threshold*100}% of peak)")
        
    elif alignment_method == 'multi_threshold':
        # Use multiple threshold points for more robust alignment
        ref_amplitude = np.max(ref_analysis_current)
        
        # Generate threshold points spread across the rising phase range
        thresholds = np.linspace(0.2, 0.8, threshold_points)
        ref_threshold_times = {}
        
        for threshold in thresholds:
            ref_threshold_value = ref_amplitude * threshold
            # Find time at which reference crosses the threshold
            thres_idx = np.argmin(np.abs(ref_analysis_current - ref_threshold_value))
            ref_threshold_times[threshold] = ref_time[thres_idx]
            
        print(f"Using {threshold_points} reference points from {thresholds[0]*100}% to {thresholds[-1]*100}% for alignment")
            
    elif alignment_method == 'curve_fit':
        # We'll need the full reference curve for curve fitting
        # Create an interpolated reference function for comparison
        from scipy.interpolate import interp1d
        ref_interp = interp1d(ref_time, ref_analysis_current, 
                             kind='linear', bounds_error=False, fill_value=(ref_analysis_current[0], ref_analysis_current[-1]))
        print(f"Created reference curve interpolation function for alignment")

    # Process each sweep
    for sweep in valid_sweep_columns:
        if sweep == reference_sweep:
            continue
            
        # Skip outliers if they should be excluded
        if exclude_outliers and trace_stats[sweep].get('is_outlier', False):
            print(f"Skipping outlier trace {sweep}")
            shifts[sweep] = np.nan
            continue

        try:
            current = data[sweep].values
            valid_indices = ~np.isnan(current) & time_mask

            # Extract valid data
            valid_time = time[valid_indices]
            valid_current = current[valid_indices]
            
            # Get trace statistics
            sweep_stats = trace_stats[sweep]
            is_negative = sweep_stats['is_negative']
            analysis_current = -valid_current if is_negative else valid_current
            
            # Find peak
            peak_idx = sweep_stats['peak_idx']
            peak_time = sweep_stats['peak_time']
            
            # If using rising phase only, restrict the data
            if rising_phase_only and peak_idx > 0:
                valid_time = valid_time[:peak_idx+1]  # Include peak
                valid_current = valid_current[:peak_idx+1]
                analysis_current = analysis_current[:peak_idx+1]

            # Apply the selected alignment method
            if alignment_method == 'single_threshold':
                # Calculate threshold value for this sweep (same % of max as reference)
                amplitude = np.max(analysis_current)
                threshold_value = amplitude * alignment_threshold

                # Find time at which this sweep crosses the threshold
                threshold_idx = np.argmin(np.abs(analysis_current - threshold_value))
                threshold_time = valid_time[threshold_idx]

                # Calculate shift needed to align with reference
                shift = ref_threshold_time - threshold_time
                
            elif alignment_method == 'multi_threshold':
                # Calculate multiple threshold crossings and average the shift
                amplitude = np.max(analysis_current)
                
                # Calculate shifts for each threshold point
                threshold_shifts = []
                
                for threshold, ref_time_point in ref_threshold_times.items():
                    try:
                        threshold_value = base_val + amplitude * threshold
                        # Find time at which this sweep crosses the threshold
                        threshold_idx = np.argmin(np.abs(analysis_current - threshold_value))
                        threshold_time = valid_time[threshold_idx]
                        
                        # Calculate this shift
                        this_shift = ref_time_point - threshold_time
                        threshold_shifts.append(this_shift)
                    except:
                        # Skip this threshold if there's a problem
                        pass
                
                if threshold_shifts:
                    # Use median shift to be more robust against outliers
                    shift = np.median(threshold_shifts)
                else:
                    # Fall back to single threshold if multi-threshold fails
                    print(f"Multi-threshold failed for {sweep}, falling back to single threshold")
                    threshold_value = base_val + amplitude * alignment_threshold
                    threshold_idx = np.argmin(np.abs(analysis_current - threshold_value))
                    threshold_time = valid_time[threshold_idx]
                    shift = ref_threshold_time - threshold_time
                    
            elif alignment_method == 'curve_fit':
                # Align by minimizing overall difference between curves
                # Create interpolated function for this sweep
                from scipy.interpolate import interp1d
                from scipy.optimize import minimize
                
                sweep_interp = interp1d(valid_time, analysis_current, 
                                      kind='linear', bounds_error=False, 
                                      fill_value=(analysis_current[0], analysis_current[-1]))
                
                # Define function to minimize (sum of squared differences)
                def shift_error(shift_value):
                    # Common time points for comparison
                    common_min = max(np.min(ref_time), np.min(valid_time) + shift_value)
                    common_max = min(np.max(ref_time), np.max(valid_time) + shift_value)
                    
                    if common_max <= common_min:
                        return 1e10  # Return large error if no overlap
                    
                    # Create comparison points
                    compare_times = np.linspace(common_min, common_max, 50)
                    
                    # Get values at aligned points
                    ref_values = ref_interp(compare_times)
                    sweep_values = sweep_interp(compare_times - shift_value)
                    
                    # Return sum of squared differences
                    return np.sum((ref_values - sweep_values) ** 2)
                
                # Find optimal shift (bounded to avoid extreme shifts)
                result = minimize(shift_error, 0.0, bounds=[(-max_shift, max_shift)], method='L-BFGS-B')
                shift = result.x[0]
            
            # Limit extreme shifts
            if abs(shift) > max_shift:
                print(f"Excessive shift detected for {sweep}: {shift:.3f} ms, limiting to ±{max_shift} ms")
                shift = np.sign(shift) * max_shift
            
            shifts[sweep] = shift

        except Exception as e:
            print(f"Error processing {sweep}: {e}")
            shifts[sweep] = np.nan

    # Count successful calculations
    success_count = sum(~np.isnan(list(shifts.values())))
    print(f"Successfully calculated shifts for {success_count} of {len(valid_sweep_columns)} sweeps")

    return shifts
    
def visualize_aligned_traces(data, shift_results, time_col='Time (ms)', max_sweeps=8,
                           x_min=0, x_max=5, highlight_rising_phase=True, auto_detect_polarity=True):
    """
    Visualize traces before and after alignment.
    Modified to better highlight the rising phase used for alignment.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing time and current traces
    shift_results : dict
        Results from the trace alignment method
    time_col : str
        Name of the time column
    max_sweeps : int
        Maximum number of sweeps to visualize
    x_min : float
        Minimum value for x-axis in plots (ms)
    x_max : float
        Maximum value for x-axis in plots (ms)
    highlight_rising_phase : bool
        If True, highlight the rising phase used for alignment
    auto_detect_polarity : bool
        If True, automatically detect negative vs positive currents
    """
    time = data[time_col].values

    # Filter for sweeps with valid shifts
    valid_sweeps = [s for s in shift_results if not np.isnan(shift_results[s])]

    if not valid_sweeps:
        print("No valid shifts to visualize")
        return None

    # Extract voltage values and sort sweeps by voltage
    voltage_pattern = re.compile(r'(-?\d+) mV')

    # Create list of (sweep_name, voltage_value) tuples
    sweep_voltage_pairs = []
    for sweep in valid_sweeps:
        match = voltage_pattern.search(sweep)
        if match:
            voltage = int(match.group(1))
            sweep_voltage_pairs.append((sweep, voltage))
        else:
            # If no voltage in name, add with a placeholder voltage
            sweep_voltage_pairs.append((sweep, 999))  # High number to sort at end

    # Sort by voltage in descending order (from least negative to most negative)
    sweep_voltage_pairs.sort(key=lambda x: x[1], reverse=True)

    # Extract the sorted sweep names
    plot_sweeps = [pair[0] for pair in sweep_voltage_pairs]

    # Limit number of sweeps if needed
    if len(plot_sweeps) > max_sweeps:
        print(f"Limiting visualization to {max_sweeps} of {len(plot_sweeps)} valid sweeps")
        # Sample evenly but preserve the voltage ordering
        indices = np.linspace(0, len(plot_sweeps)-1, max_sweeps).astype(int)
        plot_sweeps = [plot_sweeps[i] for i in indices]

    # Create figure with before/after panels stacked vertically
    plt.figure(figsize=(7.8, 8.4))  # Adjusted dimensions for vertical layout

    # Create top subplot (before alignment)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_xlabel('Time (ms)', fontweight='bold',fontsize=14)
    ax1.set_ylabel('Current (normalized)', fontweight='bold',fontsize=14)
    #ax1.set_title('Before Alignment', fontweight='bold')

    # Create bottom subplot (after alignment)
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlabel('Time (ms)', fontweight='bold',fontsize=14)
    ax2.set_ylabel('Current (normalized)', fontweight='bold',fontsize=14)
    #ax2.set_title('After Alignment', fontweight='bold')

    # Use tab10 colormap for consistency with other methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_sweeps)))

    # Find reference sweep (one with zero shift)
    ref_sweep = None
    for sweep in valid_sweeps:
        if shift_results[sweep] == 0:
            ref_sweep = sweep
            break

    # Store peak times and rising phase indices for highlighting
    peak_times = {}
    rising_phases = {}
    
    # First pass - detect all peaks for consistent highlighting
    for sweep in plot_sweeps:
        current = data[sweep].values
        valid_indices = ~np.isnan(current)
        valid_time = time[valid_indices]
        valid_current = current[valid_indices]
        
        # Check if this is a negative current and process accordingly
        if auto_detect_polarity:
            stable_segment = valid_current[-int(len(valid_current)/5):]  # Last 20%
            is_negative = np.mean(stable_segment) < 0
        else:
            is_negative = False
            
        # For analysis, if negative current, flip it
        analysis_current = -valid_current if is_negative else valid_current
        
        # Find peak for highlighting rising phase
        peak_idx = np.argmax(analysis_current)
        peak_time = valid_time[peak_idx] if peak_idx < len(valid_time) else np.nan
        peak_times[sweep] = peak_time
        
        # Store rising phase indices
        if peak_idx > 0:
            rising_phases[sweep] = np.arange(peak_idx + 1)  # Include peak point
    
    # Plot each sweep
    for i, sweep in enumerate(plot_sweeps):
        current = data[sweep].values
        valid_indices = ~np.isnan(current)
        valid_time = time[valid_indices]
        valid_current = current[valid_indices]

        if len(valid_time) < 3:
            continue
            
        # Check if this is a negative current
        if auto_detect_polarity:
            stable_segment = valid_current[-int(len(valid_current)/5):]  # Last 20%
            is_negative = np.mean(stable_segment) < 0
        else:
            is_negative = False

        # Plot original trace
        ax1.plot(valid_time, valid_current, '-',
              color=colors[i], linewidth=2,
              label=f"{sweep}" + (" (neg)" if is_negative else ""))

        # Plot shifted trace
        shift = shift_results[sweep]
        shifted_time = time + shift
        ax2.plot(shifted_time[valid_indices], valid_current, '-',
              color=colors[i], linewidth=2,
              label=f"{sweep} ({shift:.3f} ms)")
              
        # Highlight rising phase if requested
        if highlight_rising_phase and sweep in peak_times and sweep in rising_phases:
            peak_time = peak_times[sweep]
            rising_indices = rising_phases[sweep]
            
            if peak_time is not None and not np.isnan(peak_time) and len(rising_indices) > 0:
                # Original trace - highlight rising phase
                rising_mask = np.zeros_like(valid_indices, dtype=bool)
                rising_mask[rising_indices] = True
                rising_mask = rising_mask & valid_indices
                
                # Check if we have any valid points in the rising phase
                if np.any(rising_mask):
                    ax1.plot(valid_time[rising_mask], 
                          valid_current[rising_mask], '-',
                          color=colors[i], linewidth=4, alpha=0.4)
                
                    # Also highlight with vertical line at peak
                    ax1.axvline(x=peak_time, color=colors[i], linestyle='--', alpha=0.3)
                
                # Shifted trace - highlight rising phase
                shifted_rising_mask = np.zeros_like(valid_indices, dtype=bool)
                shifted_rising_mask[rising_indices] = True
                shifted_rising_mask = shifted_rising_mask & valid_indices
                
                if np.any(shifted_rising_mask):
                    ax2.plot(shifted_time[shifted_rising_mask], 
                          valid_current[shifted_rising_mask], '-',
                          color=colors[i], linewidth=4, alpha=0.4)
                    
                    # Also highlight with vertical line at shifted peak
                    shifted_peak_time = peak_time + shift
                    ax2.axvline(x=shifted_peak_time, color=colors[i], linestyle='--', alpha=0.3)

    # Auto-calculate y-limits
    y_min = min([np.min(data[sweep].dropna()) for sweep in plot_sweeps])
    y_max = max([np.max(data[sweep].dropna()) for sweep in plot_sweeps])
    
    # Add some padding
    y_range = y_max - y_min
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    # Set axis limits
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Add legends
    #ax1.legend(loc='best', fontsize=8)
    #ax2.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.show()

    # Create results table
    results_table = []
    for sweep in sorted(shift_results.keys()):
        if not np.isnan(shift_results[sweep]):
            results_table.append({
                'Sweep': sweep,
                'Shift (ms)': shift_results[sweep],
                'Is Reference': 'Yes' if sweep == ref_sweep else 'No'
            })

    if results_table:
        results_df = pd.DataFrame(results_table)
        print("\nTrace Alignment Results:")
        display(results_df)
        return results_df
    else:
        print("No valid results to display")
        return None
        
def plot_cole_moore_shifts(results_df, voltage_pattern=r'(-?\d+) mV'):
    """
    Plot Cole-Moore shifts vs. holding voltage.
    Modified to better show the relationship between voltage and shifts.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing shift results
    voltage_pattern : str
        Regex pattern to extract voltage from sweep names
    """
    # Extract voltages from sweep names
    voltages = []
    shifts = []

    for _, row in results_df.iterrows():
        if not pd.isna(row['Shift (ms)']):
            match = re.search(voltage_pattern, row['Sweep'])
            if match:
                voltage = float(match.group(1))
                voltages.append(voltage)
                shifts.append(row['Shift (ms)'])

    if len(voltages) < 2:
        print("Not enough valid voltage-shift pairs to plot")
        return

    # Sort by voltage
    voltage_shift_pairs = sorted(zip(voltages, shifts))
    voltages = [v for v, _ in voltage_shift_pairs]
    shifts = [s for _, s in voltage_shift_pairs]

    # Create figure
    plt.figure(figsize=(7.5, 8))
    
    # Plot points and connecting line
    plt.plot(voltages, shifts, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    
    # Add point labels
    for v, s in zip(voltages, shifts):
        plt.annotate(f"{v} mV", xy=(v, s), xytext=(0, 5), 
                   textcoords='offset points', ha='center', fontsize=8)
    
    # Set labels and title
    plt.xlabel('Holding Voltage (mV)', fontsize=12, fontweight='bold')
    plt.ylabel('Cole-Moore Shift (ms)', fontsize=12, fontweight='bold')
    plt.title('Cole-Moore Shift vs. Holding Voltage', fontsize=14, fontweight='bold')
    
    # Add grid and improve appearance
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)  # Zero line
    
    # Show the activation direction with an arrow annotation
    voltages_range = max(voltages) - min(voltages)
    shifts_range = max(shifts) - min(shifts)
    

    plt.tight_layout()
    plt.show()
    
    # Calculate statistics for the shifts
    if shifts:
        min_shift = min(shifts)
        max_shift = max(shifts)
        range_shift = max_shift - min_shift
        mean_shift = np.mean(shifts)
        median_shift = np.median(shifts)
        
        print("\nCole-Moore Shift Statistics:")
        print(f"Minimum Shift: {min_shift:.3f} ms at {voltages[shifts.index(min_shift)]} mV")
        print(f"Maximum Shift: {max_shift:.3f} ms at {voltages[shifts.index(max_shift)]} mV")
        print(f"Shift Range: {range_shift:.3f} ms")
        print(f"Mean Shift: {mean_shift:.3f} ms")
        print(f"Median Shift: {median_shift:.3f} ms")