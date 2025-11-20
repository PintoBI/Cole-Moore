"""
Derivative Peak Analysis Module for Cole-Moore Analysis

This module provides functions to analyze the Cole-Moore shift using derivative peak method.
It calculates derivatives of current traces, identifies their peaks or maximum values,
and extrapolates to find the activation delay time.
Modified to handle both positive and negative currents with inactivation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.signal import savgol_filter, find_peaks
from IPython.display import display

def analyze_derivative_peaks(data, time_col='Time (ms)', start_time=None,
                           window_length=11, polyorder=3, min_peak_height=None,
                           peak_prominence=0.01, handle_negative=True,
                           max_analysis_time=15.0,  # Add maximum time to consider for peak detection
                           early_activation_bias=True):  # Bias toward earlier peaks
    """
    Calculate derivatives of current traces and determine their peaks or maximum points.
    Modified to handle both positive and negative currents and limit the analysis timeframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing time and current traces
    time_col : str
        Name of the time column
    start_time : float or None
        Minimum time to consider for analysis (None to use all data)
    window_length : int
        Window length for Savitzky-Golay filter (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay filter
    min_peak_height : float or None
        Minimum height for peak detection (None for auto-detection)
    peak_prominence : float
        Minimum prominence for peak detection (helps with noisy data)
    handle_negative : bool
        If True, automatically detect negative currents and find trough instead of peak
    max_analysis_time : float
        Maximum time point to consider for analysis (helps avoid end-of-trace artifacts)
    early_activation_bias : bool
        If True, bias detection toward early activation phases

    Returns
    -------
    dict
        Dictionary with peak/max times and values for each sweep
    """
    import numpy as np
    from scipy.signal import savgol_filter, find_peaks
    
    time = data[time_col].values
    sweep_columns = [col for col in data.columns if col != time_col]

    print(f"Processing {len(sweep_columns)} sweep columns for derivative analysis")

    # If start_time is specified, create a mask for time values
    if start_time is not None:
        time_mask = time >= start_time
    else:
        time_mask = np.ones_like(time, dtype=bool)
        
    # Add mask for maximum analysis time
    if max_analysis_time is not None:
        time_mask &= time <= max_analysis_time
        print(f"Limiting analysis to time window: {start_time or 0} to {max_analysis_time} ms")

    results = {}
    for sweep in sweep_columns:
        # Skip invalid columns
        if data[sweep].isna().all() or len(data[sweep].dropna()) < window_length:
            results[sweep] = {'peak_time': np.nan, 'peak_value': np.nan, 'has_peak': False, 'polarity': 'unknown'}
            continue

        try:
            current = data[sweep].values
            valid_indices = ~np.isnan(current) & time_mask

            # Skip if not enough valid points
            if np.sum(valid_indices) < window_length:
                results[sweep] = {'peak_time': np.nan, 'peak_value': np.nan, 'has_peak': False, 'polarity': 'unknown'}
                continue

            # Extract valid data
            valid_time = time[valid_indices]
            valid_current = current[valid_indices]
            
            # Determine if we're dealing with negative currents
            # Check if the majority of the final points are negative
            end_segment = valid_current[-int(len(valid_current)/5):] # Last 20% of trace
            is_negative_current = np.mean(end_segment) < 0 if handle_negative else False
            polarity = 'negative' if is_negative_current else 'positive'
            
            # If trace appears to be a negative current, flip it for analysis
            analysis_current = -valid_current if is_negative_current else valid_current

            # Apply Savitzky-Golay filter for smoothing
            try:
                # Ensure window length is odd
                window_size = window_length if window_length % 2 == 1 else window_length + 1
                # Ensure polyorder < window_length
                poly_order = min(polyorder, window_size - 2)

                smoothed_current = savgol_filter(analysis_current, window_size, poly_order)
            except Exception as e:
                print(f"Smoothing error for {sweep}: {e}")
                # Fallback to simpler smoothing
                kernel_size = min(window_length, len(analysis_current) // 3)
                if kernel_size >= 3:
                    kernel = np.ones(kernel_size) / kernel_size
                    smoothed_current = np.convolve(analysis_current, kernel, mode='same')
                else:
                    smoothed_current = analysis_current

            # Calculate derivative
            derivative = np.gradient(smoothed_current, valid_time)
            
            # Find the activation phase by looking for rapid rise
            # This helps identify the relevant part of the trace for peak detection
            rising_phase_start = 0
            rising_phase_end = len(derivative)
            
            # Identify rough activation phase by looking for sustained increase in current
            if early_activation_bias:
                # Calculate rate of change to find fastest rising segment
                abs_derivative = np.abs(derivative)
                # Use moving average to smooth derivative
                window = 5
                if len(abs_derivative) > window:
                    smooth_abs_derivative = np.convolve(abs_derivative, np.ones(window)/window, mode='same')
                    # Find the point where the derivative starts increasing significantly
                    threshold = 0.2 * np.max(smooth_abs_derivative)
                    above_threshold = np.where(smooth_abs_derivative > threshold)[0]
                    if len(above_threshold) > 0:
                        rising_phase_start = max(0, above_threshold[0] - window)
                        # Limit analysis to a reasonable timeframe after activation starts
                        rising_phase_end = min(len(derivative), rising_phase_start + int(len(derivative) * 0.5))
                        print(f"For {sweep}, identified activation phase from {valid_time[rising_phase_start]:.2f} to {valid_time[min(rising_phase_end, len(valid_time)-1)]:.2f} ms")

            # Auto-determine min_peak_height if not provided
            # Focus on the activation phase for determining threshold
            activation_derivative = derivative[rising_phase_start:rising_phase_end]
            if len(activation_derivative) < 3:
                activation_derivative = derivative  # Fallback to full derivative
                
            if min_peak_height is None:
                # Use a percentage of the derivative's maximum value in the activation phase
                min_height = np.max(activation_derivative) * 0.3  # 30% of max
            else:
                min_height = min_peak_height

            # First try to find peaks within the activation phase
            peaks, peak_properties = find_peaks(
                activation_derivative,
                height=min_height,
                prominence=peak_prominence
            )
            
            # If no peaks found in activation phase, fall back to full range but with stricter criteria
            if len(peaks) == 0:
                peaks, peak_properties = find_peaks(
                    derivative,
                    height=min_height,
                    prominence=peak_prominence * 2  # Increase prominence requirement for full trace
                )

            # Check if we found valid peaks
            if len(peaks) > 0:
                # Preference for earlier peaks if multiple found (likely more relevant for activation)
                if early_activation_bias and len(peaks) > 1:
                    # Find earliest peak that is at least 70% as high as the highest peak
                    highest_val = np.max(derivative[peaks])
                    threshold = highest_val * 0.7
                    valid_peaks = peaks[derivative[peaks] >= threshold]
                    if len(valid_peaks) > 0:
                        # Choose earliest of the valid peaks
                        earliest_peak_idx = valid_peaks[0]
                    else:
                        # Fallback to highest peak
                        earliest_peak_idx = peaks[np.argmax(derivative[peaks])]
                    
                    peak_time = valid_time[earliest_peak_idx]
                    peak_value = derivative[earliest_peak_idx]
                else:
                    # Use highest peak (original behavior)
                    highest_peak_idx = peaks[np.argmax(derivative[peaks])]
                    peak_time = valid_time[highest_peak_idx]
                    peak_value = derivative[highest_peak_idx]
                
                # For negative currents, restore original sign for reporting but not for analysis
                if is_negative_current:
                    original_peak_value = -peak_value
                else:
                    original_peak_value = peak_value

                # Additional validation: check if peak is at the very end of the trace (likely artifact)
                if highest_peak_idx >= len(derivative) - 3:
                    print(f"Warning: Peak detected at end of trace for {sweep}, likely an artifact")
                    # Try finding an earlier peak
                    if len(peaks) > 1:
                        # Find the next highest peak
                        peaks_sorted = sorted(peaks, key=lambda p: derivative[p], reverse=True)
                        for p in peaks_sorted[1:]:  # Skip the highest peak we already found
                            if p < len(derivative) - 3:  # Not at the end
                                highest_peak_idx = p
                                peak_time = valid_time[highest_peak_idx]
                                peak_value = derivative[highest_peak_idx]
                                if is_negative_current:
                                    original_peak_value = -peak_value
                                else:
                                    original_peak_value = peak_value
                                print(f"Found alternative peak at {peak_time:.2f} ms")
                                break
                        else:
                            # No suitable alternative found
                            print(f"No suitable alternative peak found for {sweep}")
                            results[sweep] = {
                                'peak_time': np.nan,
                                'peak_value': np.nan,
                                'has_peak': False,
                                'trace_valid': True,
                                'polarity': polarity
                            }
                            continue

                # Skip if peak time is exactly at start_time (likely artifact)
                if start_time is not None and abs(peak_time - start_time) < 1e-6:
                    print(f"Skipping {sweep}: Peak detected exactly at start_time, likely an artifact")
                    results[sweep] = {
                        'peak_time': np.nan,
                        'peak_value': np.nan,
                        'has_peak': False,
                        'trace_valid': True,
                        'polarity': polarity
                    }
                    continue

                # Check if the peak is followed by a decrease in the derivative
                # (to distinguish real peaks from monotonic increases)
                has_peak = False
                if highest_peak_idx < len(derivative) - 5:  # Need points after peak
                    # Check if the derivative values after the peak are lower
                    post_peak = derivative[highest_peak_idx:highest_peak_idx+5]
                    if np.mean(post_peak) < peak_value:
                        has_peak = True

                results[sweep] = {
                    'peak_time': peak_time,
                    'peak_value': original_peak_value,
                    'has_peak': has_peak,
                    'trace_valid': True,
                    'polarity': polarity
                }
            else:
                # No peaks found, use maximum value in the activation phase
                if len(activation_derivative) > 0:
                    max_idx_local = np.argmax(activation_derivative)
                    max_idx = max_idx_local + rising_phase_start  # Convert to original index
                    max_time = valid_time[max_idx]
                    max_value = derivative[max_idx]
                else:
                    # Fall back to global maximum as last resort
                    max_idx = np.argmax(derivative)
                    max_time = valid_time[max_idx]
                    max_value = derivative[max_idx]
                
                # For negative currents, restore original sign for reporting
                if is_negative_current:
                    original_max_value = -max_value
                else:
                    original_max_value = max_value

                # Skip if max time is exactly at start_time or at the end (likely artifact)
                if (start_time is not None and abs(max_time - start_time) < 1e-6) or max_idx >= len(derivative) - 3:
                    print(f"Skipping {sweep}: Max detected at boundary, likely an artifact")
                    results[sweep] = {
                        'peak_time': np.nan,
                        'peak_value': np.nan,
                        'has_peak': False,
                        'trace_valid': True,
                        'polarity': polarity
                    }
                    continue

                results[sweep] = {
                    'peak_time': max_time,
                    'peak_value': original_max_value,
                    'has_peak': False,
                    'trace_valid': True,
                    'polarity': polarity
                }
                print(f"No clear peak found for {sweep}, using maximum derivative value at {max_time:.2f} ms")

        except Exception as e:
            print(f"Error processing {sweep}: {e}")
            results[sweep] = {
                'peak_time': np.nan,
                'peak_value': np.nan,
                'has_peak': False,
                'trace_valid': False,
                'polarity': 'unknown'
            }

    success_count = sum(not np.isnan(r.get('peak_time', np.nan)) for r in results.values())
    peak_count = sum(r.get('has_peak', False) for r in results.values())
    valid_trace_count = sum(r.get('trace_valid', False) for r in results.values())
    negative_count = sum(r.get('polarity', '') == 'negative' for r in results.values())
    positive_count = sum(r.get('polarity', '') == 'positive' for r in results.values())
    
    print(f"Successfully analyzed {success_count} of {len(sweep_columns)} sweeps")
    print(f"Found clear peaks in {peak_count} sweeps")
    print(f"Valid traces for visualization: {valid_trace_count}")
    print(f"Detected {positive_count} positive current traces and {negative_count} negative current traces")
    
    return results

def visualize_derivative_peaks(data, derivative_results, time_col='Time (ms)', max_sweeps=4,
                             window_length=11, polyorder=3, start_time=None,
                             x_min=0, x_max=5, y_min_deriv=None, y_max_deriv=None):
    """
    Visualize the derivatives and their peak points with automatic current unit scaling.
    Replaces the previous function; chooses units (A, mA, µA, nA, pA) based on the magnitude
    of the plotted currents and updates axis labels accordingly.
    """
    import numpy as np
    import matplotlib.ticker as mtick

    time = data[time_col].values
    sweep_columns = [col for col in data.columns if col != time_col]

    # Define time mask for analysis
    if start_time is not None:
        time_mask = time >= start_time
    else:
        time_mask = np.ones_like(time, dtype=bool)

    # Get all traces with valid data (regardless of peak detection)
    valid_trace_sweeps = [s for s in sweep_columns if s in derivative_results and
                        derivative_results[s].get('trace_valid', False)]

    # Get traces with valid peaks/maxima for labeling
    valid_peak_sweeps = [s for s in sweep_columns if s in derivative_results and
                       not np.isnan(derivative_results[s].get('peak_time', np.nan))]

    if not valid_trace_sweeps:
        print("No valid traces to visualize")
        return None

    # Limit the number of sweeps to plot
    if len(valid_trace_sweeps) > max_sweeps:
        # Sample evenly across the available sweeps
        indices = np.linspace(0, len(valid_trace_sweeps)-1, max_sweeps).astype(int)
        plot_sweeps = [valid_trace_sweeps[i] for i in indices]
    else:
        plot_sweeps = valid_trace_sweeps

    # Helper: choose scale factor and unit string from absolute amplitude
    def choose_current_unit(max_abs_val):
        """
        Input: max_abs_val in amps (float, >=0)
        Returns: (factor, unit_str)
        factor: divide values by factor to get numbers in that unit (e.g. for nA factor=1e-9)
        unit_str: string like 'A', 'mA', 'µA', 'nA', 'pA'
        """
        if max_abs_val == 0 or np.isnan(max_abs_val):
            return 1.0, 'A'
        # thresholds for switching units
        if max_abs_val >= 1:
            return 1.0, 'A'
        elif max_abs_val >= 1e-3:
            return 1e-3, 'mA'
        elif max_abs_val >= 1e-6:
            return 1e-6, 'µA'
        elif max_abs_val >= 1e-9:
            return 1e-9, 'nA'
        else:
            return 1e-12, 'pA'

    # Create figure with two panels
    plt.figure(figsize=(7.8, 5.6))

    # Top panel: Original traces
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_xlabel('Time (ms)', fontweight='bold', fontsize=14)
    # ylabel will be set after choosing units

    # Bottom panel: Derivatives
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlabel('Time (ms)', fontweight='bold', fontsize=14)
    # ylabel for derivative will be set after choosing units

    # Use a better color map for distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_sweeps)))

    # Collect derivative values and current values for y-axis scaling and unit selection
    all_derivatives = []
    all_currents = []

    # First pass: calculate all derivatives for proper scaling
    for i, sweep in enumerate(plot_sweeps):
        current = data[sweep].values
        valid_indices = ~np.isnan(current) & time_mask

        if np.sum(valid_indices) < 3:
            continue

        valid_time = time[valid_indices]
        valid_current = current[valid_indices]

        mask = (valid_time >= x_min) & (valid_time <= x_max)
        if np.any(mask):
            all_currents.extend(valid_current[mask])

        # Determine if this is a negative current
        is_negative = derivative_results[sweep].get('polarity', '') == 'negative'

        # For analysis, flip negative currents
        analysis_current = -valid_current if is_negative else valid_current

        # Calculate derivative
        try:
            # Apply smoothing
            window_size = min(window_length, len(analysis_current))
            if window_size % 2 == 0:
                window_size -= 1  # Ensure odd window size
            if window_size >= 3:
                smoothed = savgol_filter(analysis_current, window_size, polyorder)
            else:
                smoothed = analysis_current

            # Calculate derivative (preserve original sign for visualization)
            derivative = np.gradient(smoothed, valid_time)
            if is_negative:
                derivative = -derivative  # Flip back for display

            # Store for y-axis scaling
            mask = (valid_time >= x_min) & (valid_time <= x_max)
            if np.any(mask):
                all_derivatives.extend(derivative[mask])
        except Exception as e:
            print(f"Error calculating derivative for {sweep}: {e}")

    # Determine scaling for current axis
    if all_currents:
        max_abs_current = np.nanmax(np.abs(all_currents))
    else:
        max_abs_current = 0.0

    current_factor, current_unit = choose_current_unit(max_abs_current)
    # derivative unit factor is same current_factor, but per ms
    deriv_factor = current_factor

    # Formatters to rescale axis tick labels (divide by factor to show values in chosen unit)
    def current_formatter(x, pos):
        # Avoid dividing by zero
        return f"{(x / current_factor):.3g}"

    def deriv_formatter(x, pos):
        return f"{(x / deriv_factor):.3g}"

    current_formatter_ticker = mtick.FuncFormatter(current_formatter)
    deriv_formatter_ticker = mtick.FuncFormatter(deriv_formatter)

    # Set y-axis limits for derivative plot if not specified
    if y_min_deriv is None or y_max_deriv is None:
        if all_derivatives:
            if y_min_deriv is None:
                y_min_deriv = np.percentile(all_derivatives, 1)  # 1st percentile to avoid outliers
            if y_max_deriv is None:
                y_max_deriv = np.percentile(all_derivatives, 99)  # 99th percentile to avoid outliers
        else:
            y_min_deriv = None
            y_max_deriv = None

    # Auto-determine current y-axis limits based on data (in original A)
    if all_currents:
        y_min_curr = np.percentile(all_currents, 1)  # 1st percentile
        y_max_curr = np.percentile(all_currents, 99)  # 99th percentile

        # Add some padding
        y_range = y_max_curr - y_min_curr
        if y_range == 0:
            # small constant trace: add symmetric padding
            y_min_curr -= abs(y_max_curr) * 0.1 + 1e-12
            y_max_curr += abs(y_max_curr) * 0.1 + 1e-12
        else:
            y_min_curr -= 0.1 * y_range
            y_max_curr += 0.1 * y_range
    else:
        y_min_curr = None
        y_max_curr = None

    # Second pass: plot traces and derivatives
    for i, sweep in enumerate(plot_sweeps):
        current = data[sweep].values
        valid_indices = ~np.isnan(current) & time_mask

        if np.sum(valid_indices) < 3:
            continue

        # Get polarity info
        is_negative = derivative_results[sweep].get('polarity', '') == 'negative'
        polarity_label = " (neg)" if is_negative else ""

        # Plot original trace
        valid_time = time[valid_indices]
        valid_current = current[valid_indices]
        ax1.plot(valid_time, valid_current, '-', color=colors[i], linewidth=2,
                label=f"{sweep}")

        # Calculate and plot derivative
        try:
            # For analysis, flip negative currents
            analysis_current = -valid_current if is_negative else valid_current

            # Apply smoothing
            window_size = min(window_length, len(analysis_current))
            if window_size % 2 == 0:
                window_size -= 1  # Ensure odd window size
            if window_size >= 3:
                smoothed = savgol_filter(analysis_current, window_size, polyorder)
            else:
                smoothed = analysis_current

            # Calculate derivative (preserve original sign for visualization)
            derivative = np.gradient(smoothed, valid_time)
            if is_negative:
                derivative = -derivative  # Flip back for display

            # Plot derivative
            ax2.plot(valid_time, derivative, '-',
                   color=colors[i], linewidth=2, label=f"{sweep}{polarity_label}")

            # Mark peak/maximum point ONLY if it's not NaN
            if sweep in valid_peak_sweeps:
                peak_time = derivative_results[sweep]['peak_time']
                peak_value = derivative_results[sweep]['peak_value']
                has_peak = derivative_results[sweep]['has_peak']

                # Use different marker styles for true peaks vs maxima
                marker_style = 'o' if has_peak else 's'  # circle for peaks, square for maxima

                # Find peak index and value in our calculated derivative
                peak_idx = np.argmin(np.abs(valid_time - peak_time))
                if peak_idx < len(derivative):
                    ax2.plot(peak_time, derivative[peak_idx], marker_style, color=colors[i], markersize=8)

                    # Mark point on the original trace
                    if peak_idx < len(valid_current):
                        ax1.plot(peak_time, valid_current[peak_idx], marker_style, color=colors[i], markersize=8)
                        ax1.axvline(x=peak_time, color=colors[i], linestyle='--', alpha=0.5)

        except Exception as e:
            print(f"Error visualizing derivative for {sweep}: {e}")

    # Set axis limits
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    # Apply determined y-axis limits for raw currents (in A)
    if y_min_curr is not None and y_max_curr is not None:
        ax1.set_ylim(y_min_curr, y_max_curr)

    # Set y-axis limits for derivative plot if specified
    if y_min_deriv is not None and y_max_deriv is not None:
        ax2.set_ylim(y_min_deriv, y_max_deriv)

    # Apply tick formatters to rescale axis labels to chosen units
    ax1.yaxis.set_major_formatter(current_formatter_ticker)
    ax2.yaxis.set_major_formatter(deriv_formatter_ticker)

    # Update y-axis labels to include chosen units
    ax1.set_ylabel(f"Current ({current_unit})", fontweight='bold', fontsize=14)
    ax2.set_ylabel(f"dI/dt ({current_unit}/ms)", fontweight='bold', fontsize=14)

    # Add legends with smaller font to avoid clutter
    ax1.legend(loc='best', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()

    # Plot Cole-Moore shifts (derivative peak times) vs. voltage (unchanged)
    try:
        voltage_pattern = r'(-?\d+) mV'

        voltages = []
        peak_times = []
        marker_styles = []
        polarities = []

        for sweep in sweep_columns:
            if sweep in derivative_results and not np.isnan(derivative_results[sweep].get('peak_time', np.nan)):
                match = re.search(voltage_pattern, sweep)
                if match:
                    voltage = float(match.group(1))
                    voltages.append(voltage)
                    peak_times.append(derivative_results[sweep]['peak_time'])
                    marker_styles.append('o' if derivative_results[sweep].get('has_peak', False) else 's')
                    polarities.append(derivative_results[sweep].get('polarity', 'positive'))

        if len(voltages) >= 2:
            # Sort by voltage
            voltage_time_pairs = sorted(zip(voltages, peak_times, marker_styles, polarities))
            voltages = [v for v, _, _, _ in voltage_time_pairs]
            peak_times = [t for _, t, _, _ in voltage_time_pairs]
            marker_styles = [m for _, _, m, _ in voltage_time_pairs]
            polarities = [p for _, _, _, p in voltage_time_pairs]

            plt.figure(figsize=(10, 6))
            plt.plot(voltages, peak_times, '-', color='gray', alpha=0.5, linewidth=1)

            for i, (v, t, m, p) in enumerate(zip(voltages, peak_times, marker_styles, polarities)):
                color = 'red' if p == 'negative' else 'blue'
                marker_type = m
                if i == 0 or (i > 0 and (marker_styles[i-1] != m or polarities[i-1] != p)):
                    peak_type = "Peak" if m == 'o' else "Maximum"
                    curr_type = "Negative" if p == 'negative' else "Positive"
                    label = f"{curr_type} Current {peak_type}"
                else:
                    label = ""
                plt.plot(v, t, marker_type, color=color, markersize=8, label=label if label else "")

            plt.xlabel('Holding Voltage (mV)')
            plt.ylabel('Derivative Peak/Max Time (ms)')
            plt.title('Cole-Moore Shift via Derivative Method')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Add legend without duplicate entries
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error plotting voltage-peak time relationship: {e}")

    # Create results table
    results_table = []
    for sweep in sweep_columns:
        if sweep in derivative_results and not np.isnan(derivative_results[sweep].get('peak_time', np.nan)):
            polarity = derivative_results[sweep].get('polarity', 'positive')
            results_table.append({
                'Sweep': sweep,
                'Peak/Max Time (ms)': derivative_results[sweep]['peak_time'],
                'Peak/Max Value': derivative_results[sweep]['peak_value'],
                'True Peak': 'Yes' if derivative_results[sweep].get('has_peak', False) else 'No',
                'Current Type': 'Negative' if polarity == 'negative' else 'Positive'
            })

    if results_table:
        results_df = pd.DataFrame(results_table)
        print("\nDerivative Peak/Maximum Analysis Results:")
        display(results_df)
        return results_df
    else:
        print("No valid results to display")
        return None
