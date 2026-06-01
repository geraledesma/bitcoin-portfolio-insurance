import pandas as pd
import numpy as np


def bt_mix(r1, r2, allocator, **kwargs):
    """
    Backtests a strategy by allocating between two return streams.
    r1, r2: T x N DataFrames of returns (T = time steps, N = scenarios).
    allocator: function with signature (r1, r2, **kwargs) -> T x N weights DataFrame.
    Returns a T x N DataFrame of mixed portfolio returns.
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 must have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    return weights * r1 + (1 - weights) * r2


def terminal_values(rets):
    """
    Computes the terminal wealth from a T x N DataFrame of returns.
    Returns a Series of length N.
    """
    return (rets + 1).prod()


def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Summary statistics on terminal wealth across N scenarios.
    Returns a 1-column DataFrame indexed by stat name.
    """
    terminal_wealth = (rets + 1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (-cap + terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    return pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std": terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short": e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
