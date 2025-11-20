"""
Electrophysiology Voltage Converter Module

This module provides functions to convert sweep column headers to corresponding 
voltage values based on an initial voltage and voltage step.
"""

import re
import pandas as pd


def rename_sweep_columns_to_voltage(data, time_col='Time (ms)',
                                  initial_voltage=-120, voltage_step=10):
    """
    Rename sweep columns from sweep numbers to voltage values.

    Parameters
    ----------
    data : pandas.DataFrame
        The data with time column and sweep columns
    time_col : str
        Name of the time column (should be 'Time (ms)')
    initial_voltage : float
        The voltage (in mV) corresponding to the first sweep
    voltage_step : float
        The voltage change (in mV) between consecutive sweeps

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with renamed columns
    """
    if time_col not in data.columns:
        print(f"Error: Time column '{time_col}' not found in data")
        return data

    # Get sweep columns (all columns except time)
    sweep_columns = [col for col in data.columns if col != time_col]

    if not sweep_columns:
        print("No sweep columns found to rename")
        return data

    # Check if columns are already in voltage format
    voltage_pattern = re.compile(r'^-?\d+ mV$')
    already_voltage = all(voltage_pattern.match(col) for col in sweep_columns)

    if already_voltage:
        print("Columns are already in voltage format. No changes needed.")
        return data

    # Create a copy of the dataframe
    renamed_data = data.copy()

    # Extract sweep numbers
    sweep_numbers = []
    for col in sweep_columns:
        # Try to extract the sweep number
        try:
            # Try different patterns like "Sweep10", "sweep_10", etc.
            match = re.search(r'(\d+)', col)
            if match:
                sweep_numbers.append(int(match.group(1)))
            else:
                # If can't extract, assume sequential order
                sweep_numbers.append(len(sweep_numbers))
        except:
            # If extraction fails, use the position in the list
            sweep_numbers.append(len(sweep_numbers))

    # Create a mapping dictionary from old column names to new voltage-based names
    rename_dict = {}
    for i, col in enumerate(sweep_columns):
        # Calculate the voltage for this sweep
        # If sweep_numbers exists for this column, use it, otherwise use position
        sweep_idx = sweep_numbers[i] if i < len(sweep_numbers) else i

        voltage = initial_voltage + ((sweep_idx-1)* voltage_step)
        new_col_name = f"{voltage} mV"
        rename_dict[col] = new_col_name

    # Rename the columns
    renamed_data = renamed_data.rename(columns=rename_dict)

    print(f"Column renaming complete. Changed {len(rename_dict)} columns.")
    print("Old → New column mapping:")
    for old, new in rename_dict.items():
        print(f"  {old} → {new}")

    return renamed_data
