import pandas as pd
import numpy as np
from scipy.optimize import minimize


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights.
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the volatility of a portfolio from a covariance matrix and weights.
    """
    return (weights.T @ covmat @ weights) ** 0.5


def minimize_vol(target_return, er, cov):
    """
    Returns optimal weights achieving the target return with minimum volatility.
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda w, er: target_return - portfolio_return(w, er)
    }
    result = minimize(
        portfolio_vol, init_guess, args=(cov,), method='SLSQP',
        options={'disp': False},
        constraints=(weights_sum_to_1, return_is_target),
        bounds=bounds
    )
    return result.x


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the Maximum Sharpe Ratio portfolio.
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    def neg_sharpe(w, riskfree_rate, er, cov):
        r = portfolio_return(w, er)
        vol = portfolio_vol(w, cov)
        return -(r - riskfree_rate) / vol

    result = minimize(
        neg_sharpe, init_guess,
        args=(riskfree_rate, er, cov), method='SLSQP',
        options={'disp': False},
        constraints=(weights_sum_to_1,),
        bounds=bounds
    )
    return result.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio.
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    """
    Returns a grid of n_points weights spanning the efficient frontier.
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    return [minimize_vol(target_return, er, cov) for target_return in target_rs]


def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier.
    """
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=".-")


def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False,
            riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier.
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        ax.plot([0, vol_msr], [riskfree_rate, r_msr],
                color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1 / n, n)
        ax.plot([portfolio_vol(w_ew, cov)], [portfolio_return(w_ew, er)],
                color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        ax.plot([portfolio_vol(w_gmv, cov)], [portfolio_return(w_gmv, er)],
                color='midnightblue', marker='o', markersize=10)
        return ax
