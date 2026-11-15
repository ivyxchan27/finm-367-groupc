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


# ---------------------------------------------------------------------------
# Forecasting SPY helpers
# ---------------------------------------------------------------------------
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_ols():
    return LinearRegression()


def make_rf():
    return RandomForestRegressor(max_depth=5, n_estimators=100, random_state=0, n_jobs=-1)


def make_nn():
    """MLP with standardized features and target; predictions come back in the original scale.
    The scalers are refit on every .fit(), so OOS each step only sees data through t."""
    net = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=0)
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), net),
                                      transformer=StandardScaler())


def lag_align(y, X):
    """Lag signals one period so row t+1 holds x_t, aligned with the return r_{t+1} it forecasts."""
    X_lag = X.shift().dropna()
    y = y.reindex(X_lag.index).dropna()
    return y, X_lag.loc[y.index]


def expanding_mean_forecast(y):
    """Null forecast: the mean of returns through t, dated t+1."""
    return y.expanding().mean().shift().dropna()


def insample_forecasts(y, X_lag, specs):
    """Full-sample fitted values. specs maps name -> (signal columns, model factory)."""
    out = {}
    for name, (cols, make_model) in specs.items():
        model = make_model().fit(X_lag[cols], y)
        out[name] = pd.Series(model.predict(X_lag[cols]), index=y.index)
    return pd.DataFrame(out)


def oos_forecast(y, X_lag, X, cols, make_model, burn):
    """Expanding-window forecasts: fit on data through t, predict r_{t+1} from x_t, date it t+1."""
    preds = {}
    for i in range(burn - 1, len(y)):
        t = y.index[i]
        model = make_model().fit(X_lag.loc[:t, cols], y.loc[:t])
        preds[t] = model.predict(X.loc[[t], cols])[0]
    # shift so each forecast is stamped with the month it forecasts; the final one is beyond the sample
    return pd.Series(preds).shift().dropna()


def oos_forecasts(y, X_lag, X, specs, burn):
    return pd.DataFrame({name: oos_forecast(y, X_lag, X, cols, make_model, burn)
                         for name, (cols, make_model) in specs.items()})


def strategy_returns(fcst, y, scale=100):
    """w_t = scale * forecast of r_{t+1}; strategy return w_t * r_{t+1}. Adds a Passive (w=1) column."""
    strats = fcst.mul(scale).mul(y, axis=0).dropna()
    strats.insert(0, 'Passive', y.loc[strats.index])
    return strats


def r2_insample(y, fcst):
    """Usual R^2 of fitted values against the full-sample mean of y."""
    y = y.loc[fcst.index]
    sst = ((y - y.mean()) ** 2).sum()
    return fcst.apply(lambda f: 1 - ((y - f) ** 2).sum() / sst).rename('R-squared')


def r2_oos(y, fcst, null):
    """1 - MSE(forecast) / MSE(null), over the dates both are available."""
    idx = fcst.dropna().index.intersection(null.dropna().index)
    mse_null = ((y.loc[idx] - null.loc[idx]) ** 2).mean()
    return fcst.loc[idx].apply(lambda f: 1 - ((y.loc[idx] - f) ** 2).mean() / mse_null).rename('OOS R-squared')


def plot_cumulative(rets, title):
    ax = (1 + rets).cumprod().plot(logy=True, figsize=(10, 5), title=title)
    ax.set_ylabel('Growth of $1 (log scale)')
    ax.set_xlabel('')
    return ax
