# Cole–Moore Analysis Package

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PintoBI/Cole-Moore/blob/main/Cole_Moore_colab.ipynb)

**Quick start:** run the full analysis pipeline interactively using the Google Colab notebook above. No local installation required.

---

## Overview

This repository provides tools to **quantify activation delay and the Cole–Moore shift** from macroscopic ionic currents.  
It implements three  analysis methods. This package introduces a **derivative-based metric** that allows objective, reproducible quantification of activation delay, while also providing traditional fitting- and alignment-based approaches for comparison.

---

## Implemented Methods

### 1. Derivative Peak Method (recommended)
- Obtains the time of the maximum of `dI/dt`
- Robust in the prescence of inactivation and multiple kinetic components
- Minimal assumptions about current trace shape

### 2. Exponential Extrapolation
- Fits single, double and triple exponentials
- Estimates activation delay by extrapolation to baseline

### 3. Trace Alignment
- Threshold-based alignment of current traces
- Useful for relative comparisons across prepulses

---

## Google Colab Notebook

The repository includes a fully documented tutorial notebook:

📓 **Cole_Moore_colab.ipynb**

It demonstrates:
- data loading and preprocessing
- application of all three methods
- visualization and method comparison
- export of results

Open it directly here:  
https://colab.research.google.com/github/PintoBI/Cole-Moore/blob/main/Cole_Moore_colab.ipynb

---

## Local Installation

```bash
git clone https://github.com/PintoBI/Cole-Moore.git
cd Cole-Moore
```

## Citation

If you use this package in your research, please cite:

> **Pinto-Anwandter, B. I. & Bezanilla, F.**  
> *Quantifying Activation Delay and the Cole–Moore Shift via Current Derivatives*  
> **Biophysical Journal** (2025)  
> https://www.sciencedirect.com/science/article/pii/](https://www.sciencedirect.com/science/article/pii/S000634952503468X
