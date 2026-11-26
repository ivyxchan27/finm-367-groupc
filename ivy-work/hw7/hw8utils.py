import numpy as np
import pandas as pd
import statsmodels.api as sm


def summary_stats(rets, ann = 12, quantile = 0.05):
    """
    Annualized mean, vol and Sharpe, plus skewness, excess kurtosis and a lower quantile.

    Args:
    - rets (pd.DataFrame): periodic returns, one column per series
    - ann (int): periods per year used to annualize (12 for monthly data)
    - quantile (float): lower quantile to report

    Returns (pd.DataFrame): one row per series, one column per statistic
    """
    out = pd.DataFrame(index = rets.columns)
    out["Mean"] = rets.mean() * ann
    out["Vol"] = rets.std() * np.sqrt(ann)
    out["Sharpe"] = out["Mean"] / out["Vol"]
    out["Skewness"] = rets.skew()
    out["Excess Kurtosis"] = rets.kurtosis()
    out[f"{quantile:.0%} Quantile"] = rets.quantile(quantile)
    return out


def lfd(y, X, ann = 12):
    """
    Linear factor decomposition: OLS of y on X with an intercept.

    Args:
    - y (pd.Series): returns being explained
    - X (pd.Series or pd.DataFrame): factor returns, one column per factor
    - ann (int): periods per year used to annualize (12 for monthly data)

    Returns (pd.Series): annualized alpha, one beta per factor, t(alpha), r-squared and info ratio
    """
    X = X.to_frame() if isinstance(X, pd.Series) else X
    df = pd.concat([y, X], axis = 1).dropna()
    model = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])).fit()
    out = pd.Series({"Alpha": model.params["const"] * ann})
    out = pd.concat([out, model.params.drop("const").add_prefix("Beta ")])
    out["t(Alpha)"] = model.tvalues["const"]
    out["R-squared"] = model.rsquared
    out["Info Ratio"] = model.params["const"] * ann / (model.resid.std() * np.sqrt(ann))
    return out


def nested_f_test(y, X_restricted, X_full):
    """
    F-test of whether the extra factors in a larger regression improve on a nested smaller one.

    Args:
    - y (pd.Series): returns being explained
    - X_restricted (pd.Series or pd.DataFrame): factors in the smaller model
    - X_full (pd.DataFrame): factors in the larger model (must include those in X_restricted)

    Returns (pd.Series): F-statistic, p-value and the number of restrictions tested
    """
    X_restricted = X_restricted.to_frame() if isinstance(X_restricted, pd.Series) else X_restricted
    df = pd.concat([y, X_full], axis = 1).dropna()
    restricted = sm.OLS(df.iloc[:, 0], sm.add_constant(df[X_restricted.columns])).fit()
    full = sm.OLS(df.iloc[:, 0], sm.add_constant(df[X_full.columns])).fit()
    f_stat, p_value, df_diff = full.compare_f_test(restricted)
    return pd.Series({"F-statistic": f_stat, "p-value": p_value, "Restrictions": df_diff})


def captioned(df, caption, precision = 4):
    """
    Attach a bold caption above a table so it renders as part of the table in the notebook.

    Args:
    - df (pd.DataFrame): table to display
    - caption (str): caption shown above the table, e.g. "Table 2.7: ..."
    - precision (int): decimal places shown for float columns

    Returns (pandas.io.formats.style.Styler): styled table with the caption
    """
    return (df.style
            .set_caption(caption)
            .format(precision = precision)
            .set_table_styles([{"selector": "caption", "props": "caption-side: top; font-weight: bold; font-size: 1.1em; text-align: left; padding-bottom: 6px;"}]))
