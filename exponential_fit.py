
"""
Exponential Fitting Module for Cole-Moore Analysis (generalized to 1, 2, or 3 exponentials)

This module fits sums of exponentials to current traces and extrapolates to the
time where the fitted curve crosses 0 ("zero-crossing") for Cole–Moore analysis.

What's new (vs. the original):
- Generalized N-exponential model (N = 1, 2, or 3) with the *activation* form:
    y(t) = C + sum_i A_i * (1 - exp(-t / tau_i))
- Unified parameter initialization, bounds, fitting, RMSE, and reporting
- Optional selection of N via exp_type in {'single','double','triple'} or directly with n_exp=1..3
- Visualization handles single/double/triple consistently
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from IPython.display import display


# ------------------------------
#   Model & utilities (generic)
# ------------------------------
def exp_sum_activation(t, *params):
    """
    Activation-form sum of exponentials:
        y(t) = C + sum_{i=1..n} A_i * (1 - exp(-t/tau_i))
    params = [A1, tau1, A2, tau2, ..., An, taun, C]
    """
    n = (len(params) - 1) // 2
    C = params[-1]
    out = np.full_like(t, C, dtype=float)
    for i in range(n):
        A = params[2*i]
        tau = max(params[2*i + 1], 1e-12)
        out += A * (1.0 - np.exp(-t / tau))
    return out


def _initial_guesses_n(n, t, y):
    """
    Build robust initial guesses for N components.
    Uses span-based tau guesses and distributes amplitude among components.
    """
    # Rough amplitude and baseline
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    amp = ymax - ymin if np.isfinite(ymax - ymin) and (ymax - ymin) != 0 else 1.0
    C0 = np.median(y[-max(10, len(y)//10):]) if len(y) > 0 else 0.0

    span = max(np.nanmax(t) - np.nanmin(t), 1e-3)

    if n == 1:
        taus = [max(0.05, span/3.0)]
        amps = [amp]
    elif n == 2:
        taus = [max(0.1, span/10.0), max(0.1, span/2.0)]
        amps = [0.6*amp, 0.4*amp]
    else:  # n == 3
        taus = [max(0.1, span/40.0), max(0.1, span/5.0), max(0.1, span/2.0)]
        amps = [0.5*amp, 0.3*amp, 0.2*amp]

    p0 = []
    for A, tau in zip(amps, taus):
        p0 += [A, tau]
    p0 += [C0]
    return np.array(p0, dtype=float)


def _bounds_n(n):
    """
    Bounds for A_i, tau_i, and C:
      A_i free (±inf), tau_i in [1e-6, 1e6], C free (±inf)
    """
    lower, upper = [], []
    for _ in range(n):
        lower += [-np.inf, 1e-6]
        upper += [ np.inf, 1e6]
    lower += [-np.inf]  # C
    upper += [ np.inf]
    return (np.array(lower, float), np.array(upper, float))


def _fit_nexp(n, t, y):
    p0 = _initial_guesses_n(n, t, y)
    bounds = _bounds_n(n)
    try:
        popt, pcov = curve_fit(
            lambda tt, *pp: exp_sum_activation(tt, *pp),
            t, y, p0=p0, bounds=bounds, maxfev=50000
        )
        yfit = exp_sum_activation(t, *popt)
        resid = y - yfit
        rmse = float(np.sqrt(np.mean(resid**2)))
        return popt, pcov, yfit, resid, rmse
    except Exception as e:
        return None, None, None, None, np.inf


def _zero_crossing_time_from_fit(params, tmin, tmax, num=2000):
    """
    Find t where the fitted curve crosses 0. Uses a broad search window
    based on taus present in 'params'.
    """
    if params is None:
        return np.nan

    n = (len(params) - 1) // 2
    taus = [max(params[2*i+1], 1e-12) for i in range(n)]
    tau_min = min(taus) if taus else 1.0
    tau_max = max(taus) if taus else 1.0

    # Expand a reasonable window around [tmin, tmax]
    t_lo = max(0.0, tmin - 5*tau_max)
    t_hi = tmax + 15*tau_max
    tt = np.linspace(t_lo, t_hi, num=num)
    yy = exp_sum_activation(tt, *params)

    # Look for sign changes
    sgn = np.sign(yy)
    crossings = np.where(np.diff(np.signbit(yy)))[0]
    if len(crossings) == 0:
        return np.nan

    # First crossing
    i0 = crossings[0]
    # Linear interpolate for better estimate
    x0, x1 = tt[i0], tt[i0+1]
    y0, y1 = yy[i0], yy[i0+1]
    if (y1 - y0) != 0:
        t_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
    else:
        t_cross = x0
    return float(t_cross)


# --------------------------------------------------
#   Analyze with N-exponential fitting
# --------------------------------------------------
def analyze_with_exponential_fit(
    data,
    time_col='Time (ms)',
    start_time=None,
    end_time=None,
    baseline_percent=10,
    max_percent=90,
    exp_type='single',
    n_exp=None,
):
    """
    Fit N-exponential activation model to each sweep and compute zero crossing.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with time column and one or more sweep columns
    time_col : str
        Name of the time column
    start_time, end_time : float or None
        Optional time window for fitting; if None, use entire range
    baseline_percent, max_percent : float
        Percent thresholds (relative to sweep min->max) to select the fitting segment
    exp_type : {'single','double','triple'}
        Convenience selector for N; ignored if n_exp is given
    n_exp : int or None
        If provided, must be 1, 2, or 3; overrides exp_type

    Returns
    -------
    dict
        results[sweep] = {
            'crossing_time': float,
            'taus': [tau1, tau2, ...],
            'params': [A1, tau1, A2, tau2, ..., C],
            'rmse': float,
            'component_percents': [p1, p2, ...]  # based on |A_i|
        }
    """
    if n_exp is None:
        mapping = {'single': 1, 'double': 2, 'triple': 3}
        n_exp = mapping.get(str(exp_type).lower(), 1)
    n_exp = int(n_exp)
    if n_exp not in (1, 2, 3):
        raise ValueError("n_exp must be 1, 2, or 3")

    time = data[time_col].values
    sweep_columns = [c for c in data.columns if c != time_col]
    print(f"Processing {len(sweep_columns)} sweep columns with {n_exp}-exponential fitting")

    # Global mask for time window used to *choose points* for the fit
    time_mask = np.ones_like(time, dtype=bool)
    if start_time is not None:
        time_mask &= (time >= start_time)
    if end_time is not None:
        time_mask &= (time <= end_time)

    results = {}

    for sweep in sweep_columns:
        cur = data[sweep].values
        valid = (~np.isnan(cur)) & time_mask
        if np.sum(valid) < 5:
            results[sweep] = {'crossing_time': np.nan, 'taus': [np.nan]*n_exp,
                              'params': [np.nan]*(2*n_exp+1), 'rmse': np.nan,
                              'component_percents': [np.nan]*n_exp}
            print(f"Skipping {sweep}: not enough points.")
            continue

        # Threshold the fitting region based on % of (min..max) within the selected time window
        fit_y = cur[valid]
        fit_t = time[valid]
        ymin, ymax = np.min(fit_y), np.max(fit_y)
        amplitude = ymax - ymin
        if amplitude < 1e-12:
            results[sweep] = {'crossing_time': np.nan, 'taus': [np.nan]*n_exp,
                              'params': [np.nan]*(2*n_exp+1), 'rmse': np.nan,
                              'component_percents': [np.nan]*n_exp}
            print(f"Skipping {sweep}: amplitude too small.")
            continue

        low_thr = ymin + amplitude * (baseline_percent / 100.0)
        hi_thr  = ymin + amplitude * (max_percent / 100.0)
        use = valid & (cur >= low_thr) & (cur <= hi_thr)
        if np.sum(use) < 5:
            results[sweep] = {'crossing_time': np.nan, 'taus': [np.nan]*n_exp,
                              'params': [np.nan]*(2*n_exp+1), 'rmse': np.nan,
                              'component_percents': [np.nan]*n_exp}
            print(f"Skipping {sweep}: not enough points within thresholds.")
            continue

        T = time[use]
        Y = cur[use]

        # Fit n-exp model on the selected window
        popt, pcov, yfit, resid, rmse = _fit_nexp(n_exp, T, Y)

        if popt is None:
            results[sweep] = {'crossing_time': np.nan, 'taus': [np.nan]*n_exp,
                              'params': [np.nan]*(2*n_exp+1), 'rmse': np.nan,
                              'component_percents': [np.nan]*n_exp}
            print(f"Fit failed for {sweep}")
            continue

        # Extract taus and component percentages (by |A_i|)
        taus = [max(popt[2*i+1], 1e-12) for i in range(n_exp)]
        amps = [popt[2*i] for i in range(n_exp)]
        denom = sum(abs(a) for a in amps)
        perc = [(abs(a)/denom*100.0) if denom > 0 else np.nan for a in amps]

        # Estimate zero-crossing from the fitted function over an extended window
        tmin = np.nanmin(time) if len(time) else 0.0
        tmax = np.nanmax(time) if len(time) else max(T) if len(T) else 1.0
        t0 = _zero_crossing_time_from_fit(popt, tmin, tmax)

        results[sweep] = {
            'crossing_time': t0,
            'taus': taus,
            'params': popt,
            'rmse': rmse,
            'component_percents': perc,
        }

    ok = sum(1 for k, v in results.items() if np.isfinite(v.get('rmse', np.nan)))
    print(f"Finished: {ok} of {len(sweep_columns)} sweeps produced fits.")
    return results


# --------------------------------------------------
#   Visualization (supports 1/2/3 exponentials)
# --------------------------------------------------
def visualize_exponential_fits(
    data,
    exp_results,
    time_col='Time (ms)',
    max_sweeps=8,
    x_min=0.0,
    x_max=None,
):
    """
    Visualize raw traces, their fits (using stored params), and zero-crossings.
    Works for any of 1/2/3 exponentials produced by analyze_with_exponential_fit.
    """
    time = data[time_col].values
    sweep_columns = [c for c in data.columns if c != time_col]

    valid_sweeps = [s for s in sweep_columns
                    if s in exp_results and np.isfinite(exp_results[s].get('crossing_time', np.nan))]

    if not valid_sweeps:
        print("No valid exponential fits to visualize")
        return None

    if len(valid_sweeps) > max_sweeps:
        idx = np.linspace(0, len(valid_sweeps)-1, max_sweeps).astype(int)
        plot_sweeps = [valid_sweeps[i] for i in idx]
        print(f"Showing {len(plot_sweeps)} of {len(valid_sweeps)} valid sweeps")
    else:
        plot_sweeps = valid_sweeps

    # Determine axes
    max_t0 = max(exp_results[s]['crossing_time'] for s in valid_sweeps if np.isfinite(exp_results[s]['crossing_time']))
    x_max_zoom = 1.2*max_t0 if np.isfinite(max_t0) and max_t0 > 0 else np.nanmax(time)

    if x_max is None:
        x_max_upper = x_max_zoom
    else:
        x_max_upper = x_max

    extended_time = np.linspace(0.0, max(x_max_upper, x_max_zoom) * 1.2, 1200)

    # Figure
    plt.figure(figsize=(7.8, 5.6))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)

    ax1.set_xlabel('Time (ms)', fontweight='bold',size=14)
    ax1.set_ylabel('Current (normalized)', fontweight='bold',size=14)
    ax2.set_xlabel('Time (ms)', fontweight='bold',size=14)
    ax2.set_ylabel('Current (normalized)', fontweight='bold',size=14)
    ax2.axhline(0.0, linewidth=1.0)

    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_sweeps)))

    # Dashed zero line on the residual/fit panel
    ax2.axhline(0.0, linestyle='--', linewidth=1.0)

    # Plot each sweep
    for i, sweep in enumerate(plot_sweeps):
        cur = data[sweep].values
        params = exp_results[sweep]['params']

        # raw
        ax1.plot(time, cur, '-', linewidth=2, color=colors[i])
        # fit
        yfit_ext = exp_sum_activation(extended_time, *params)
        taus = exp_results[sweep]['taus']
        tau_str = "/".join([f"{x:.2g}" for x in taus])
        t0 = exp_results[sweep]['crossing_time']

        ax1.plot(extended_time, yfit_ext, '--', linewidth=2, color=colors[i])

        ax2.plot(extended_time, yfit_ext, '-', linewidth=2, color=colors[i])

        # Mark crossing and add a small vertical dashed tick
        if np.isfinite(t0) and (0 <= t0 <= x_max_zoom):
            # X marker at (t0, 0)
            ax2.plot(t0, 0.0, 'x', color=colors[i], markersize=9, markeredgewidth=2)
            # Small vertical dashed tick just below zero
            ax2.plot([t0, t0], [-0.01, -0.05], '--', color=colors[i], linewidth=2)

    # Keep the same x-lims
    ax1.set_xlim(x_min, x_max_upper)
    ax2.set_xlim(0.0, x_max_zoom)


    # Keep some comfortable y range
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 0.55)

    plt.tight_layout()
    plt.show()

    # Summarize results table
    rows = []
    for s in sweep_columns:
        r = exp_results[s]
        row = {
            'Sweep': s,
            'Crossing Time (ms)': r['crossing_time'],
            'RMSE': r['rmse'],
        }
        # up to 3 taus and component %
        for i, tau in enumerate(r['taus']):
            row[f'Tau{i+1} (ms)'] = tau
        for i, p in enumerate(r['component_percents']):
            row[f'Component{i+1} (%)'] = p
        rows.append(row)

    df = pd.DataFrame(rows)
    display(df)
    return df


# ------------------------------
#   Convenience wrappers 
# ------------------------------
def analyze_single(data, **kwargs):
    return analyze_with_exponential_fit(data, n_exp=1, **kwargs)

def analyze_double(data, **kwargs):
    return analyze_with_exponential_fit(data, n_exp=2, **kwargs)

def analyze_triple(data, **kwargs):
    return analyze_with_exponential_fit(data, n_exp=3, **kwargs)
