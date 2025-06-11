# Cole-Moore Analysis Package

A comprehensive Python package for analyzing ion channel activation delay with a focus on Cole-Moore shift.

Run this code in Google Colab

## Overview

This toolkit implements three methods to measure activation delay:

1. **Derivative Peak Analysis** - Identifies activation delay by finding peaks in current derivatives
2. **Exponential Fitting** - Fits exponential functions and extrapolates to find zero-crossing times  
3. **Trace Alignment** - Aligns traces using threshold-based methods to measure relative delays

## Installation

Clone this repository and run in Google Colab:

```python
!git clone https://github.com/pintobi/Cole-Moore
```

## Quick Start

### 1. Load your data

```python
from data_loader import load_data

# Load data from uploaded file (Google Colab)
data = load_data('your_data.txt', file_content=uploaded_content)

# Or load from file path
data = load_data('path/to/your/data.txt')
```

### 2. Run Cole-Moore analysis

```python
from exponential_fit import analyze_with_exponential_fit
from trace_aligner import align_traces
from derivative_analysis import analyze_with_derivative

# Exponential fitting method
exp_results = analyze_with_exponential_fit(data, exp_type='single')

# Trace alignment method
alignment_results = align_traces(data, alignment_method='multi_threshold')

# Derivative peak method
derivative_results = analyze_with_derivative(data)
```

### 3. Compare methods

```python
from compare_methods import compare_cole_moore_methods

# Compare all three methods
compare_cole_moore_methods(
    exp_csv='exponential_results.csv',
    shift_csv='alignment_results.csv', 
    derivative_csv='derivative_results.csv'
)
```

## Module Overview

### `data_loader.py`
- Imports data
- Converts time units (seconds to milliseconds)
- Handles various delimiters and header configurations
- Validates and cleans electrophysiology data

### `exponential_fit.py`
- Fits single or double exponential functions to current traces
- Extrapolates to find zero-crossing times
- Configurable fitting parameters and time windows

### `trace_aligner.py`
- Aligns traces based on a reference trace
- Multiple threshold alignment settings

### `derivative_analysis.py`
- Analyzes derivative peaks in current traces
- Finds maximum derivative
- Handles both positive and negative current traces

### `compare_methods.py`
- Compares results from different analysis methods
- Statistical summary of each method
- Produces comparison plots

### `cole_moore_correlation.py`
- Creates correlation plots between methods
- Calculates Pearson correlation coefficients

## Example Data

The package includes example electrophysiology datasets:

- `example/.txt` - Sample ion channel recordings

## Requirements

- Python 3.6+
- NumPy
- Pandas  
- Matplotlib
- SciPy
- IPython (for Colab display)

## File Format Support

The package automatically handles:
- Tab-delimited files (.txt, .tsv)
- Comma-separated files (.csv)
- Space-delimited files
- Various header configurations
- Time units in seconds or milliseconds

## Output

Each analysis method generates:
- CSV files with detailed results
- Summary statistics
- Comparison plots
- Correlation analysis


## Citation

If you use this package in your research, please cite:

```
TBD
```
