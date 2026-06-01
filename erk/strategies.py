import pandas as pd
import numpy as np


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Static allocation: constant weight w1 in r1 across all time steps and scenarios.
    Returns a T x N DataFrame of weights.
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)


def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
    """
    Linearly decreases allocation to r1 from start_glide to end_glide over time.
    Returns a T x N DataFrame of weights.
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path] * n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths


def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    CPPI-style allocator using a liability floor defined by zero-coupon bond prices.
    Allocates a multiple of the cushion (account - PV of floor) into the PSP.
    Returns a T x N DataFrame of PSP weights.
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor * zc_prices.iloc[step]
        cushion = (account_value - floor_value) / account_value
        psp_w = (m * cushion).clip(0, 1)
        ghp_w = 1 - psp_w
        account_value = (account_value * psp_w * (1 + psp_r.iloc[step]) +
                         account_value * ghp_w * (1 + ghp_r.iloc[step]))
        w_history.iloc[step] = psp_w
    return w_history


def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8,
             riskfree_rate=0.03, drawdown=None):
    """
    Backtests the CPPI strategy on a single risky asset return series.
    Returns a dict with wealth history, allocation history, and parameters.
    """
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = account_value

    if isinstance(risky_r, pd.Series):
        risky_r = risky_r.to_frame(name="R")
    if safe_r is None:
        safe_r = pd.DataFrame(
            data=riskfree_rate / 12,
            index=risky_r.index,
            columns=risky_r.columns
        )

    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = np.clip(m * cushion, 0, 1)
        safe_w = 1 - risky_w
        account_value = (account_value * risky_w * (1 + risky_r.iloc[step]) +
                         account_value * safe_w * (1 + safe_r.iloc[step]))
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak

    risky_wealth = start * (1 + risky_r).cumprod()
    return {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floorval_history,
        "risky_r": risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
    }


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    CPPI-style allocator that tracks the running peak and constrains
    drawdown to at most maxdd from peak. Core strategy of this project.
    Returns a T x N DataFrame of PSP weights.
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1 - maxdd) * peak_value
        cushion = (account_value - floor_value) / account_value
        psp_w = (m * cushion).clip(0, 1)
        ghp_w = 1 - psp_w
        account_value = (account_value * psp_w * (1 + psp_r.iloc[step]) +
                         account_value * ghp_w * (1 + ghp_r.iloc[step]))
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history
