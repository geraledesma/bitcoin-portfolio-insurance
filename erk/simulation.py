import math
import pandas as pd
import numpy as np

from .fixed_income import ann_to_inst, inst_to_ann


def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15,
        steps_per_year=12, s_0=100.0, prices=True):
    """
    Geometric Brownian Motion Monte Carlo simulation.
    Uses (1+mu)^dt as the per-step drift to avoid discretization bias.
    Returns a DataFrame of shape (n_steps+1, n_scenarios); row 0 is s_0.
    """
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(
        loc=(1 + mu) ** dt,
        scale=sigma * np.sqrt(dt),
        size=(n_steps, n_scenarios)
    )
    rets_plus_1[0] = 1
    ret_val = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1
    return ret_val


def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,
        steps_per_year=12, r_0=None):
    """
    Cox-Ingersoll-Ross interest rate model.
    Generates random interest rate paths and zero-coupon bond prices.
    b and r_0 are annualized rates; returned values are annualized as well.
    """
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1 / steps_per_year
    num_steps = int(n_years * steps_per_year) + 1

    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    h = math.sqrt(a**2 + 2 * sigma**2)
    prices = np.empty_like(shock)

    def price(ttm, r):
        _A = ((2 * h * math.exp((h + a) * ttm / 2)) /
              (2 * h + (h + a) * (math.exp(h * ttm) - 1))) ** (2 * a * b / sigma**2)
        _B = (2 * (math.exp(h * ttm) - 1)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
        return _A * np.exp(-_B * r)

    prices[0] = price(n_years, r_0)

    for step in range(1, num_steps):
        r_t = rates[step - 1]
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        rates[step] = abs(r_t + d_r_t)
        prices[step] = price(n_years - step * dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    return rates, prices
