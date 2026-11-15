import numpy as np
import pandas as pd
import statsmodels.api as sm


def max_drawdown(rets):
    wealth = (1 + rets).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def summary_stats(rets, ann=52):
    out = pd.DataFrame(index=rets.columns)
    out['Mean'] = rets.mean() * ann
    out['Vol'] = rets.std() * np.sqrt(ann)
    out['Sharpe'] = out['Mean'] / out['Vol']
    out['Min'] = rets.min()
    out['5th Pct'] = rets.quantile(0.05)
    out['Max Drawdown'] = rets.apply(lambda s: max_drawdown(s.dropna()))
    return out


def lfd(y, X, ann=52):
    """OLS of y on X (with intercept). Returns annualized alpha, betas, r-squared, info ratio."""
    X = X.to_frame() if isinstance(X, pd.Series) else X
    df = pd.concat([y, X], axis=1).dropna()
    model = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])).fit()
    out = pd.Series({'Alpha': model.params['const'] * ann})
    out = pd.concat([out, model.params.drop('const')])
    out['t(Alpha)'] = model.tvalues['const']
    out['R-squared'] = model.rsquared
    out['Info Ratio'] = model.params['const'] * ann / (model.resid.std() * np.sqrt(ann))
    return out


def lfd_table(xrets, fund, factors, ann=52):
    """Multivariate LFD of one fund on a list of factors, over their common sample."""
    df = xrets[[fund] + factors].dropna()
    print(f'{fund} sample: {df.index[0].date()} to {df.index[-1].date()}')
    return lfd(df[fund], df[factors], ann).to_frame(fund).T


def rolling_lfd(xrets, fund, factors, window, ann=52):
    """Rolling-window LFD; returns annualized alpha and betas indexed by each window's end date."""
    df = xrets[[fund] + factors].dropna()
    rolling = {}
    for end in range(window, len(df) + 1):
        win = df.iloc[end - window:end]
        model = sm.OLS(win[fund], sm.add_constant(win[factors])).fit()
        rolling[win.index[-1]] = model.params.rename({'const': 'Alpha'})
    betas = pd.DataFrame(rolling).T
    betas['Alpha'] *= ann
    return betas


def period_summary(r, ann=52):
    """Start/end dates, cumulative and annualized return of a single return series."""
    r = r.dropna()
    years = len(r) / ann
    total = (1 + r).prod() - 1
    return pd.Series({'Start': r.index[0].date(), 'End': r.index[-1].date(),
                      'Cumulative Return': total, 'Annualized': (1 + total) ** (1 / years) - 1})
