import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def performanceMetrics(returns, annualization=1, quantile=0.05):
    metrics = pd.DataFrame(index=returns.columns)
    metrics['Mean'] = returns.mean() * annualization
    metrics['Vol'] = returns.std() * np.sqrt(annualization)
    metrics['Sharpe'] = metrics['Mean'] / metrics['Vol']
    metrics['Min'] = returns.min()
    metrics['Max'] = returns.max()
    return metrics


def maximumDrawdown(returns):
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max

    max_drawdown = drawdown.min()
    end_date = drawdown.idxmin()
    summary = pd.DataFrame({'Max Drawdown': max_drawdown, 'Bottom': end_date})

    for col in drawdown:
        summary.loc[col, 'Peak'] = rolling_max.loc[:end_date[col], col].idxmax()
        recovery = drawdown.loc[end_date[col]:, col]
        recovered = recovery[recovery >= 0]
        summary.loc[col, 'Recovery'] = recovered.index[0] if len(recovered) else pd.NaT
        summary.loc[col, 'Duration (to Recover)'] = (
            summary.loc[col, 'Recovery'] - summary.loc[col, 'Peak']
            if len(recovered) else pd.NaT
        )

    return summary[['Max Drawdown', 'Peak', 'Bottom', 'Recovery', 'Duration (to Recover)']]


def tailMetrics(returns, quantile=0.05, mdd=True):
    metrics = pd.DataFrame(index=returns.columns)
    metrics['Skewness'] = returns.skew()
    metrics['Kurtosis'] = returns.kurtosis()

    VaR = returns.quantile(quantile)
    CVaR = returns[returns <= VaR].mean()
    metrics[f'VaR ({quantile})'] = VaR
    metrics[f'CVaR ({quantile})'] = CVaR

    if mdd:
        mdd_stats = maximumDrawdown(returns)
        metrics = metrics.join(mdd_stats)

    return metrics


def get_ols_metrics(regressors, targets, annualization=1, ignorenan=True):
    if isinstance(regressors, pd.Series):
        regressors = regressors.to_frame()
    if isinstance(targets, pd.Series):
        targets = targets.to_frame()

    df_out = pd.DataFrame(index=targets.columns)

    for col in targets.columns:
        y = targets[col]
        X = regressors.reindex(y.index)
        if ignorenan:
            mask = y.notna() & X.notna().all(axis=1)
            y, X = y[mask], X[mask]

        reg = LinearRegression().fit(X, y)
        df_out.loc[col, 'alpha'] = reg.intercept_ * annualization
        for i, name in enumerate(X.columns):
            df_out.loc[col, name] = reg.coef_[i]
        df_out.loc[col, 'r-squared'] = reg.score(X, y)

        if X.shape[1] == 1:
            vol_resid = (y - reg.predict(X)).std()
            df_out.loc[col, 'Treynor Ratio'] = (y.mean() / reg.coef_[0]) * annualization
            df_out.loc[col, 'Info Ratio'] = (reg.intercept_ / vol_resid) * np.sqrt(annualization)

    return df_out
