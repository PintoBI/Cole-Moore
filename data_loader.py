"""
Electrophysiology Data Loader Module

This module provides functions to load and process electrophysiology data files
with automatic detection of time units, delimiters, and header lines.
"""

import numpy as np
import pandas as pd
import re  # For regular expressions (used in time unit detection)
import io  # For handling file content as a file stream


def load_data(filename, file_content=None, header_lines=2, time_col_hint='Time', delimiter=None):
    """
    Load electrophysiology data, handling time unit conversion (s to ms).

    Parameters
    ----------
    filename : str
        Path or description of the data file.
    file_content : bytes or None
        Content of the file as bytes (from Colab upload).
    header_lines : int
        Initial guess for number of header lines to skip.
    time_col_hint : str
        A hint for identifying the time column (e.g., 'Time', 'ms', 'sec').
    delimiter : str or None
        Delimiter for parsing (None to auto-detect).

    Returns
    -------
    data : pandas.DataFrame or None
        Loaded data with 'Time (ms)' column and current traces, or None if loading fails.
    """
    if file_content is None:
        try:
            # Try reading with latin-1 for broader compatibility if reading from path
            with open(filename, 'r', errors='latin-1') as f:
                file_content_str = f.read()
            print(f"Reading data from file path: {filename}")
        except FileNotFoundError:
            print(f"Error: File not found at path {filename}. Please upload the file.")
            return None
        except Exception as e:
            print(f"Error reading file from path {filename}: {e}")
            return None
    else:
        # Decode uploaded file content
        try:
            file_content_str = file_content.decode('utf-8')
            print(f"Reading data from uploaded file: {filename} (UTF-8)")
        except UnicodeDecodeError:
            try:
                file_content_str = file_content.decode('latin-1')
                print(f"Reading data from uploaded file: {filename} (Latin-1)")
            except Exception as e:
                print(f"Error decoding file content for {filename}: {e}")
                return None
        except Exception as e:
            print(f"Error processing file content for {filename}: {e}")
            return None

    lines = file_content_str.splitlines()
    if not lines:
        print(f"Error: File {filename} is empty.")
        return None

    print(f"File '{filename}' has {len(lines)} lines")

    # --- Find Header and Delimiter ---
    header_line_index = -1
    potential_time_hints = [time_col_hint.lower(), 'time', 'ms', 'sec']
    for i in range(min(20, len(lines))):  # Check more lines for header
        line_content = lines[i].strip()
        # Skip empty/commented lines, look for time hint and likely delimiters
        if not line_content or line_content.startswith(('#', '%')):
            continue
        line_lower = line_content.lower()
        if any(hint in line_lower for hint in potential_time_hints) and \
           (',' in line_content or '\t' in line_content or re.search(r'\s{2,}', line_content)):
            header_line_index = i
            print(f"Found potential header at line {i+1}: {lines[i].strip()[:80]}...")
            break

    if header_line_index == -1:
        print(f"Warning: Could not reliably find header line containing '{time_col_hint}'. Using specified header_lines={header_lines}.")
        header_line_index = header_lines
        if header_line_index >= len(lines):
            print(f"Error: header_lines ({header_lines}) is too large for the file length ({len(lines)}).")
            return None
    header_lines_to_skip = header_line_index

    # Detect delimiter
    if delimiter is None:
        header_text = lines[header_line_index].strip()
        if '\t' in header_text:
            delimiter = '\t'
            print("Detected tab delimiter in header.")
        elif ',' in header_text:
            delimiter = ','
            print("Detected comma delimiter in header.")
        else:  # Check data lines
            delimiters_found = {}
            for i in range(header_line_index + 1, min(header_line_index + 6, len(lines))):
                if '\t' in lines[i]:
                    delimiters_found['\t'] = delimiters_found.get('\t', 0) + 1
                if ',' in lines[i]:
                    delimiters_found[','] = delimiters_found.get(',', 0) + 1
                if re.search(r'\s{2,}', lines[i].strip()):
                    delimiters_found[r'\s+'] = delimiters_found.get(r'\s+', 0) + 1
            if delimiters_found:
                delimiter = max(delimiters_found, key=delimiters_found.get)
                print(f"Detected delimiter '{delimiter}' in data lines.")
            else:
                delimiter = r'\s+'
                print("No specific delimiter detected, trying whitespace.")

    # --- Parse Data ---
    file_io = io.StringIO(file_content_str)
    data = None
    parse_attempts = [delimiter] + [d for d in ['\t', ',', r'\s+'] if d != delimiter]  # Try specified delimiter first

    for attempt_delimiter in parse_attempts:
        try:
            print(f"Attempting to parse with delimiter='{attempt_delimiter}' and skiprows={header_lines_to_skip}")
            file_io.seek(0)
            data = pd.read_csv(file_io, skiprows=header_lines_to_skip, sep=attempt_delimiter, engine='python',
                             on_bad_lines='warn', skipinitialspace=True)
            # Check if parsing looks reasonable (multiple columns, first column has data)
            if data.shape[1] > 1 and data.iloc[:, 0].notna().sum() > 1:
                print(f"Successfully parsed with delimiter='{attempt_delimiter}'")
                break  # Stop trying if successful
            else:
                data = None  # Reset if parsing seems incorrect
        except Exception as e:
            data = None
        if data is None:
            print(f"Parsing failed or resulted in single/empty column with delimiter='{attempt_delimiter}'.")

    if data is None:
        print(f"Failed to parse file '{filename}'. Please check header_lines and delimiter parameters.")
        return None

    # --- Data Cleaning ---
    data.columns = [str(col).strip() for col in data.columns]
    # Rename duplicate columns
    cols = pd.Series(data.columns)
    dup_indices = cols[cols.duplicated()].index
    for idx in dup_indices:
        cols[idx] = f"{cols[idx]}_{idx}"  # Simple renaming with index
    data.columns = cols
    print(f"Cleaned columns: {data.columns.tolist()}")

    # Convert to numeric, coerce errors
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(axis=1, how='all', inplace=True)  # Drop fully empty columns

    # --- Identify and Convert Time Column ---
    found_time_col_name = None
    time_col_is_seconds = False
    matching_cols = [col for col in data.columns if time_col_hint.lower() in col.lower()]
    if matching_cols:
        found_time_col_name = matching_cols[0]
        print(f"Found time column matching hint '{time_col_hint}': '{found_time_col_name}'")
    else:  # Broader search
        for col in data.columns:
            if any(hint in col.lower() for hint in ['time', 'ms', 'sec', '(s)']):
                found_time_col_name = col
                print(f"Found potential time column: '{found_time_col_name}'")
                break
        if not found_time_col_name and len(data.columns) > 0 and data.iloc[:, 0].notna().any():
            found_time_col_name = data.columns[0]
            print(f"Warning: Using first column '{found_time_col_name}' as time.")
    if not found_time_col_name:
        print("Error: Could not identify a time column.")
        return None

    # Check specifically for "Time ( s)" for seconds, "Time ( ms)" for milliseconds
    if "time ( s)" in found_time_col_name.lower() or "time (s)" in found_time_col_name.lower():
        time_col_is_seconds = True
        print(f"Detected time unit as seconds in '{found_time_col_name}'. Converting to milliseconds.")
        data[found_time_col_name] = data[found_time_col_name] * 1000.0
    elif "time ( ms)" in found_time_col_name.lower() or "time (ms)" in found_time_col_name.lower():
        print(f"Detected time unit as milliseconds in '{found_time_col_name}'. No conversion needed.")
    else:
        # Fallback to more general detection
        if re.search(r'(\bs\b)|(\(s\))|second', found_time_col_name.lower()):
            time_col_is_seconds = True
            print(f"Detected time unit as seconds in '{found_time_col_name}'. Converting to milliseconds.")
            data[found_time_col_name] = data[found_time_col_name] * 1000.0
        else:
            print(f"Assuming time unit is milliseconds for '{found_time_col_name}'.")

    # Rename to standard 'Time (ms)'
    standard_time_col = 'Time (ms)'
    if found_time_col_name != standard_time_col:
        if standard_time_col in data.columns:
            data = data.rename(columns={standard_time_col: standard_time_col + "_original"})
        data = data.rename(columns={found_time_col_name: standard_time_col})
        print(f"Renamed time column to '{standard_time_col}'.")
    time_col = standard_time_col

    # --- Final Validation ---
    if time_col not in data.columns:
        print(f"Error: Standard time column '{time_col}' not found.")
        return None
    if data[time_col].isna().any():
        print(f"Warning: Dropping {data[time_col].isna().sum()} rows with NaN time.")
        data = data.dropna(subset=[time_col])
        if data.empty:
            print("Error: Data empty after dropping NaN time rows.")
            return None

    # Identify valid sweep columns
    sweep_columns = [col for col in data.columns if col != time_col]
    valid_sweep_columns = [col for col in sweep_columns if data[col].notna().sum() > 3]
    if not valid_sweep_columns:
        print("Error: No valid sweep columns found.")
        return None

    final_data_columns = [time_col] + valid_sweep_columns
    data = data[final_data_columns].copy()  # Return a copy
    print(f"Found {len(valid_sweep_columns)} valid sweep columns: {valid_sweep_columns}")
    print(f"Data loaded successfully. Shape: {data.shape}")

    return data
