"""
risk_assessment.py
──────────────────
Comprehensive risk analytics for individual stocks and portfolios.
Metrics: Volatility, Sharpe, Sortino, MaxDD, VaR, CVaR, Beta, Calmar.
"""

import os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs")


def compute_risk_metrics(returns: pd.Series, rf_annual: float = 0.06,
                         name: str = "Asset") -> dict:
    """Compute full risk metric suite for a return series."""
    daily_rf = (1 + rf_annual) ** (1/252) - 1
    excess = returns - daily_rf

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = (ann_ret - rf_annual) / (ann_vol + 1e-10)

    # Sortino ratio (downside deviation)
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = (ann_ret - rf_annual) / (downside_std + 1e-10)

    # Maximum drawdown
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = ann_ret / (abs(max_dd) + 1e-10)

    # VaR and CVaR at 95%
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    # Skewness and Kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()

    # Win rate
    win_rate = (returns > 0).mean()

    return {
        "Name": name,
        "Ann_Return": round(ann_ret, 4),
        "Ann_Volatility": round(ann_vol, 4),
        "Sharpe_Ratio": round(sharpe, 4),
        "Sortino_Ratio": round(sortino, 4),
        "Max_Drawdown": round(max_dd, 4),
        "Calmar_Ratio": round(calmar, 4),
        "VaR_95": round(var_95, 6),
        "CVaR_95": round(cvar_95, 6),
        "Skewness": round(skew, 4),
        "Kurtosis": round(kurt, 4),
        "Win_Rate": round(win_rate, 4),
    }


def compute_portfolio_risk(returns_df: pd.DataFrame, weights: dict,
                           name: str = "Portfolio") -> dict:
    """Risk metrics for a weighted portfolio."""
    syms = list(weights.keys())
    avail = [s for s in syms if s in returns_df.columns]
    w = np.array([weights[s] for s in avail])
    w /= w.sum()

    port_returns = (returns_df[avail] * w).sum(axis=1)
    return compute_risk_metrics(port_returns, name=name)


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """CAPM beta."""
    cov = np.cov(stock_returns.dropna(), market_returns.dropna())
    return cov[0, 1] / (cov[1, 1] + 1e-10)


def full_risk_report(df: pd.DataFrame):
    """Generate risk report for all stocks and save."""
    # Build daily returns matrix
    prices = df.pivot_table(index="Date", columns="Symbol", values="Close")
    returns = prices.pct_change()  # keep NaN per stock, don't dropna across all

    # Market proxy: equal-weighted NIFTY-50 (use available stocks each day)
    market = returns.mean(axis=1)

    risk_data = []
    for sym in returns.columns:
        sr = returns[sym].dropna()
        if len(sr) < 100:
            continue
        metrics = compute_risk_metrics(sr, name=sym)
        aligned = pd.concat([sr, market], axis=1).dropna()
        if len(aligned) > 50:
            metrics["Beta"] = round(compute_beta(aligned.iloc[:, 0], aligned.iloc[:, 1]), 4)
        else:
            metrics["Beta"] = np.nan
        risk_data.append(metrics)

    risk_df = pd.DataFrame(risk_data)
    risk_df.to_csv(os.path.join(OUTPUT_DIR, "risk_report.csv"), index=False)
    print(f"Risk report saved: {len(risk_df)} stocks")
    return risk_df


if __name__ == "__main__":
    data_path = os.path.join(PROJ_ROOT, "data", "processed", "nifty50_features.parquet")
    df = pd.read_parquet(data_path)
    risk_df = full_risk_report(df)
    print("\nTop 10 by Sharpe Ratio:")
    print(risk_df.sort_values("Sharpe_Ratio", ascending=False).head(10)[
        ["Name", "Ann_Return", "Ann_Volatility", "Sharpe_Ratio", "Max_Drawdown", "Beta"]
    ].to_string(index=False))
