import math
from scipy.stats import t, tvar, norm
from scipy.optimize import minimize
import numpy as np
from scipy import stats
import pandas as pd
from numpy.linalg import eig
from typing import Union, List, Callable
from dataclasses import dataclass
from scipy.stats import rv_continuous


class FittedModel:
    def __init__(self, beta, error_model, eval_func, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval_func = eval_func
        self.errors = errors
        self.u = u

    beta: Union[List[float], None]
    error_model: rv_continuous
    eval_func: Callable
    errors: List[float]
    u: List[float]


def VaR(a, alpha=0.05):
    x = np.sort(a)
    upperbound = int(np.ceil(len(a) * alpha)) - 1
    lowerbound = int(np.floor(len(a) * alpha)) - 1
    v = 0.5 * (x[upperbound] + x[lowerbound])
    return -v


def ES(a, alpha=0.05):
    x = np.sort(a)
    upperbound = int(np.ceil(len(a) * alpha))
    lowerbound = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[upperbound] + x[lowerbound])
    es = np.mean(x[x <= v])
    return -es


def fit_general_normal(data):
    mu, std = norm.fit(data)
    uc = norm.cdf(x=data, loc=mu, scale=std)
    # create the error model
    errorModel = np.random.normal(loc=mu, scale=std)
    # calculate the errors and U
    errors = data - mu
    u = norm.cdf(loc=mu, scale=std, x=data)

    def eval(u_value):
        return norm.ppf(q=u_value, loc=mu, scale=std)

    return [mu, std], FittedModel(None, errorModel, eval, errors, u)


def fit_general_t(data):
    
    tdf, tloc, tscale = t.fit(data)
    
    
    # def likelihood_t(params):
    #     loc, scale, df = params
    #     log_likelihood = np.sum(np.log(stats.t.pdf(x=data, loc=loc, scale=scale, df=df)))
    #     return -log_likelihood
    # 
    # start_loc = np.mean(data)
    # start_df = 6.0 / stats.kurtosis(data) + 4
    # start_scale = np.sqrt(tvar(data) * (start_df - 2) / start_df)
    # start_df = start_scale
    # start_scale = 1
    # positive_inf = float('inf')
    # negative_inf = float('-inf')
    # bnds = ((negative_inf, positive_inf), (1.0 * math.e ** -8, positive_inf), (2.0001, positive_inf))
    # tresult = minimize(likelihood_t, np.array([start_loc, start_scale, start_df]), method='L-BFGS-B',
    #                    bounds=bnds)
    # # loc, scale, df
    # tloc = tresult.x[0]
    # tscale = tresult.x[1]
    # tdf = tresult.x[2]
    # create the error model
    errorModel = t.rvs(df=tdf) * tscale + tloc
    # calculate the errors and U
    errors = data - tloc
    u = t.cdf(loc=tloc, scale=tscale, df=tdf, x=data)

    def eval(u_value):
        return t.ppf(loc=tloc, scale=tscale, df=tdf, q=u_value)

    # return tresult.x, FittedModel(None, errorModel, eval, errors, u)
    return [tloc, tscale, tdf], FittedModel(None, errorModel, eval, errors, u)


def exponential_covariance(x, lam):
    # x is m by n (m-1 is t)
    # calculate the weight matrix
    m = len(x)
    w = np.zeros(shape=(m, 1))
    # remove mean from series
    col_means = np.mean(x, axis=0)
    x = x - col_means
    for i in range(0, m):
        w[i] = (1 - lam) * (math.pow(lam, m - 1 - i))
    wsum = np.sum(w)
    wnorm = w / wsum
    return np.dot((np.transpose(wnorm * x)), x)


def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    vars = prices.columns.tolist()
    nVars = len(vars)
    vars = [var for var in vars if var != dateColumn]
    if nVars == len(vars):
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
    nVars -= 1
    pricesarr = prices[vars].values
    n, m = pricesarr.shape
    r = np.empty((n - 1, m))

    def calculate_returns(prices, r):
        for i in range(n - 1):
            for j in range(m):
                r[i, j] = prices[i + 1, j] / prices[i, j]
        return r

    r = calculate_returns(pricesarr, r)
    if method.upper() == "DISCRETE":
        r -= 1.0
    elif method.upper() == "LOG":
        r = np.log(r)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    dates = prices.iloc[1:, prices.columns.get_loc(dateColumn)]
    out = pd.DataFrame({dateColumn: dates})
    for i, var in enumerate(vars):
        out[var] = r[:, i]
    return out


def simulatePCA(a, nsim, pctExp=1, mean=[], seed=1234):
    n = a.shape[0]
    _mean = np.zeros(n)
    if mean:
        _mean = mean.copy()
    # Eigenvalue decomposition
    vals, vecs = eig(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    flip = np.arange(vals.size - 1, -1, -1)
    vals = vals[flip]
    vecs = vecs[:, flip]
    tv = np.sum(vals)
    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        nval = 0
        pct = 0.0
        # figure out how many factors we need for the requested percent explained
        for i in range(posv.size):
            pct += vals[i] / tv
            nval += 1
            if pct >= pctExp:
                break
        if nval < posv.size:
            posv = posv[:nval]
    vals = vals[posv]
    vecs = vecs[:, posv]
    B = vecs @ np.diag(np.sqrt(vals))
    np.random.seed(seed)
    m = vals.size
    r = np.random.randn(m, nsim)
    out = (B @ r).T
    # Loop over iterations and add the mean
    for i in range(n):
        out[:, i] += _mean[i]
    return out


def computePrices(values, currentStocksPrices, simReturns):
    nVals = len(values)
    currentValues = np.empty(nVals)
    simulatedValues = np.empty(nVals)
    pnls = np.empty(nVals)
    for i in range(nVals):
        price = currentStocksPrices[values['Stock'][i]]  # get current Stock price
        currentValues[i] = values['Holding'][i] * price  # holding * stockPrice
        simulatedValues[i] = currentValues[i] * (1.0 + simReturns.at[values['iteration'][i] - 1, values['Stock'][i]])
        pnls[i] = simulatedValues[i] - currentValues[i]
    values['currentValue'] = currentValues
    values['simulatedValue'] = simulatedValues
    values['pnl'] = pnls
    return values
