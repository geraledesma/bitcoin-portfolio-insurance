import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm


def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    Returns a DataFrame with columns for the wealth index,
    the previous peaks, and the percentage drawdown.
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Previous Peak": previous_peaks,
        "Drawdown": drawdowns
    })


def skewness(r):
    """
    Computes the skewness of the supplied Series or DataFrame.
    Returns a float or a Series.
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3


def kurtosis(r):
    """
    Computes the kurtosis of the supplied Series or DataFrame.
    Returns a float or a Series.
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of a Series or DataFrame.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame.
    If modified=True, uses the Cornish-Fisher modification.
    """
    z = norm.ppf(level / 100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2 - 1) * s / 6 +
             (z**3 - 3 * z) * (k - 3) / 24 -
             (2 * z**3 - 5 * z) * (s**2) / 36)
    return -(r.mean() + z * r.std(ddof=0))


def semideviation(r):
    """
    Returns the negative semideviation of r.
    r must be a Series or DataFrame.
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def compound(r):
    """
    Returns the result of compounding the set of returns in r.
    """
    return np.expm1(np.log1p(r).sum())


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal.
    Returns True if normality is accepted at the given level.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level
