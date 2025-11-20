"""
Electrophysiology Data Normalization Module

This module provides functions to normalize electrophysiology current traces
based on specified time windows and methods. Modified to handle both inward (negative)
and outward (positive) currents properly.
"""

import numpy as np
import pandas as pd


def normalize_data_window(data, time_col='Time (ms)', method='subtract',
                          start_time_ms=None, end_time_ms=None,
                          auto_detect_polarity=True, force_negative=False):
    """
    Normalize current traces based on a specified time window.
    Modified to handle both positive and negative currents.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame with a time column and sweep columns.
    time_col : str
        Name of the time column (should be 'Time (ms)').
    method : str
        Normalization method:
        'subtract': Subtract the mean of the window.
        'divide': Scale trace using max magnitude.
        'normalize': Scale trace 0-1 (or -1-0 for negative) based on min/max.
        'absolute_normalize': Scale to 0-1 regardless of polarity.
        'polarity_preserve': Normalize while preserving positive/negative direction.
        'none': Return the original data unmodified.
    start_time_ms : float or None
        Start time (in ms) of the window for calculating baseline/min-max.
        If None, uses the start of the data.
    end_time_ms : float or None
        End time (in ms) of the window for calculating baseline/min-max.
        If None, uses the end of the data.
    auto_detect_polarity : bool
        If True, automatically detect if each trace has negative (inward) currents
    force_negative : bool
        If True, assume all data represents negative (inward) currents

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with normalized sweep data, or the original if method is 'none'.
    """
    if method.lower() == 'none':
        print("Normalization skipped as method is 'none'.")
        return data.copy()  # Return a copy

    if time_col not in data.columns:
        print(f"Error: Time column '{time_col}' not found in data. Cannot normalize.")
        return data.copy()  # Return original data copy

    normalized_data = data.copy()
    time = data[time_col].values
    sweep_columns = [col for col in data.columns if col != time_col]

    print(f"Applying '{method}' normalization using window [{start_time_ms}, {end_time_ms}] ms.")
    
    if auto_detect_polarity:
        print("Automatic polarity detection enabled")
    if force_negative:
        print("Forcing negative current normalization for all traces")

    # Define the time window mask for normalization calculation
    norm_time_mask = np.ones_like(time, dtype=bool)
    if start_time_ms is not None:
        norm_time_mask &= (time >= start_time_ms)
    if end_time_ms is not None:
        norm_time_mask &= (time <= end_time_ms)

    if np.sum(norm_time_mask) == 0:
        print(f"Warning: No data points found in the specified normalization window [{start_time_ms}, {end_time_ms}]. Returning original data.")
        return data.copy()

    polarities = {}  # Store the detected polarity of each sweep
    
    # First pass - detect polarities if needed
    if auto_detect_polarity and not force_negative:
        for sweep in sweep_columns:
            current = data[sweep].values
            valid_indices_sweep = ~np.isnan(current)
            
            # Get the data in the latter part of the trace (where current stabilizes)
            if np.sum(valid_indices_sweep) > 10:
                stable_segment = current[valid_indices_sweep][-int(len(current[valid_indices_sweep])/3):]
                is_negative = np.mean(stable_segment) < 0
                polarities[sweep] = 'negative' if is_negative else 'positive'
            else:
                polarities[sweep] = 'positive'  # Default if not enough points
    
    # Second pass - normalize each sweep
    for sweep in sweep_columns:
        current = data[sweep].values
        valid_indices_sweep = ~np.isnan(current)

        # Indices valid for normalization calculation (must be valid sweep point AND in time window)
        norm_calc_indices = valid_indices_sweep & norm_time_mask

        if np.sum(norm_calc_indices) < 2:
            print(f"Warning: Skipping normalization for {sweep}: < 2 points in window [{start_time_ms}, {end_time_ms}].")
            continue  # Keep original data for this sweep

        # Determine polarity for this sweep
        if force_negative:
            is_negative = True
        elif auto_detect_polarity:
            is_negative = polarities.get(sweep, 'positive') == 'negative'
        else:
            is_negative = False
            
        polarity_label = "negative" if is_negative else "positive"
        norm_current_segment = current[norm_calc_indices]

        if method.lower() == 'subtract':
            # Define a 0.1 ms window starting from start_time_ms for baseline
            baseline_window_start = start_time_ms if start_time_ms is not None else time[0]
            baseline_window_end = baseline_window_start + 0.1  # 0.1 ms window

            # Create mask for this specific baseline window
            baseline_mask = (time >= baseline_window_start) & (time <= baseline_window_end) & valid_indices_sweep

            # Check if we have enough points in the baseline window
            if np.sum(baseline_mask) >= 3:  # Requiring at least 3 points for a stable mean
                baseline = np.mean(current[baseline_mask])
                if not np.isnan(baseline):
                    normalized_data[sweep] = current - baseline  # Apply to the whole trace
                    print(f"  {sweep}: Baseline subtracted ({baseline:.4e}) - {polarity_label} current")
                else:
                    print(f"Warning: Baseline calculation resulted in NaN for {sweep}. Skipping.")
            else:
                print(f"Warning: Not enough points in 0.1 ms baseline window for {sweep} (found {np.sum(baseline_mask)}). Skipping.")

        elif method.lower() == 'normalize':
            min_val = np.nanmin(norm_current_segment)
            max_val = np.nanmax(norm_current_segment)
            amplitude = max_val - min_val
            
            if amplitude > 1e-12:
                if is_negative:
                    # For negative currents, normalize to range -1 to 0
                    # min_val (most negative) becomes -1, 0 stays 0
                    if min_val < 0:
                        normalized_data[sweep] = current / abs(min_val)
                        print(f"  {sweep}: Normalized to [-1,0] scale (min={min_val:.4e})")
                    else:
                        print(f"  {sweep}: Warning - detected as negative but values are positive. Using standard normalization.")
                        normalized_data[sweep] = (current - min_val) / amplitude
                else:
                    # For positive currents, normalize to range 0 to 1
                    normalized_data[sweep] = (current - min_val) / amplitude
                    print(f"  {sweep}: Normalized to [0,1] scale (min={min_val:.4e}, max={max_val:.4e})")
            else:
                print(f"Warning: Skipping 'normalize' for {sweep}: Amplitude near zero ({amplitude:.2e}) in window.")
                normalized_data[sweep] = 0.0  # Set to zero if amplitude is tiny

        elif method.lower() == 'absolute_normalize':
            # Normalize to 0-1 scale regardless of polarity, based on absolute magnitude
            abs_max = np.max(np.abs(norm_current_segment))
            if abs_max > 1e-12:
                normalized_data[sweep] = np.abs(current) / abs_max
                print(f"  {sweep}: Absolute normalized with max magnitude {abs_max:.4e} - {polarity_label} current")
            else:
                print(f"Warning: Skipping 'absolute_normalize' for {sweep}: Max magnitude near zero.")
                normalized_data[sweep] = 0.0
                
        elif method.lower() == 'polarity_preserve':
            # Preserve polarity but normalize to max magnitude of 1
            abs_max = np.max(np.abs(norm_current_segment)) 
            if abs_max > 1e-12:
                normalized_data[sweep] = current / abs_max
                print(f"  {sweep}: Polarity preserved, normalized with max magnitude {abs_max:.4e} - {polarity_label} current")
            else:
                print(f"Warning: Skipping 'polarity_preserve' for {sweep}: Max magnitude near zero.")
                normalized_data[sweep] = 0.0

        elif method.lower() == 'divide':
            if is_negative:
                # For negative currents, find the most negative value (minimum)
                peak_val = np.nanmin(norm_current_segment)
                if peak_val < -1e-12:  # Ensure it's significantly negative
                    normalized_data[sweep] = current / peak_val  # Values will be positive after division
                    print(f"  {sweep}: Scaled by negative peak ({peak_val:.4e}) - negative current")
                else:
                    print(f"Warning: Skipping 'divide' for {sweep}: Minimum value not negative enough.")
                    normalized_data[sweep] = 0.0
            else:
                # For positive currents, use maximum
                max_val = np.nanmax(norm_current_segment)
                if max_val > 1e-12:
                    normalized_data[sweep] = current / max_val
                    print(f"  {sweep}: Scaled by max ({max_val:.4e}) - positive current")
                else:
                    print(f"Warning: Skipping 'divide' for {sweep}: Maximum near zero.")
                    normalized_data[sweep] = 0.0

        else:
            print(f"Warning: Unknown normalization method '{method}'. Skipping sweep {sweep}.")

    print("Normalization applied.")
    return normalized_data
