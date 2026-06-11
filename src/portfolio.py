"""
portfolio.py
────────────
Construct portfolios for Conservative, Balanced, and Aggressive investors.
Uses Mean-Variance optimization (Markowitz) with constraints per risk profile.
"""

import os, warnings
import pandas as pd
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Risk profiles ────────────────────────────────────────────────────────────
PROFILES = {
    "Conservative": {
        "target_vol": 0.15,        # annualized vol cap
        "max_single_stock": 0.10,  # max weight per stock
        "min_stocks": 15,
        "prefer_low_vol": True,
        "risk_aversion": 5.0,
    },
    "Balanced": {
        "target_vol": 0.22,
        "max_single_stock": 0.12,
        "min_stocks": 10,
        "prefer_low_vol": False,
        "risk_aversion": 2.0,
    },
    "Aggressive": {
        "target_vol": 0.35,
        "max_single_stock": 0.15,
        "min_stocks": 8,
        "prefer_low_vol": False,
        "risk_aversion": 0.5,
    },
}


def compute_stock_metrics(df: pd.DataFrame, lookback_start: str = "2019-01-01",
                          lookback_end: str = "2020-12-31") -> pd.DataFrame:
    """Compute annualized return, volatility, Sharpe, etc. for each stock."""
    sub = df[(df["Date"] >= lookback_start) & (df["Date"] <= lookback_end)].copy()

    # Daily returns
    returns = sub.pivot_table(index="Date", columns="Symbol", values="Close").pct_change().dropna()

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-10)

    # Maximum drawdown per stock
    prices = sub.pivot_table(index="Date", columns="Symbol", values="Close")
    cum_max = prices.cummax()
    drawdown = (prices - cum_max) / (cum_max + 1e-10)
    max_dd = drawdown.min()

    metrics = pd.DataFrame({
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }).dropna()

    # Get industry mapping
    meta = sub[["Symbol", "Industry"]].drop_duplicates().set_index("Symbol")
    metrics = metrics.join(meta)

    return metrics, returns


def mean_variance_optimize(mu: np.ndarray, cov: np.ndarray, risk_aversion: float,
                           max_weight: float, n_stocks: int) -> np.ndarray:
    """Optimize portfolio weights using mean-variance utility: max (mu'w - λ/2 w'Σw)."""
    n = len(mu)

    def neg_utility(w):
        ret = w @ mu
        risk = w @ cov @ w
        return -(ret - risk_aversion / 2 * risk)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # fully invested
    ]
    bounds = [(0, max_weight)] * n

    # Try multiple random initializations
    best_w = None
    best_util = -np.inf
    for _ in range(10):
        w0 = np.random.dirichlet(np.ones(n))
        w0 = np.clip(w0, 0, max_weight)
        w0 /= w0.sum()
        res = minimize(neg_utility, w0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 1000})
        if res.success and -res.fun > best_util:
            best_util = -res.fun
            best_w = res.x

    if best_w is None:
        best_w = np.ones(n) / n

    # Zero out tiny weights
    best_w[best_w < 0.005] = 0
    best_w /= best_w.sum()
    return best_w


def build_portfolio(metrics: pd.DataFrame, returns: pd.DataFrame,
                    profile_name: str) -> pd.DataFrame:
    """Build portfolio for a given risk profile."""
    profile = PROFILES[profile_name]

    # Pre-filter: for conservative, prefer low-vol stocks
    if profile["prefer_low_vol"]:
        vol_thresh = metrics["ann_volatility"].quantile(0.6)
        eligible = metrics[metrics["ann_volatility"] <= vol_thresh]
    else:
        eligible = metrics.copy()

    # Ensure enough stocks
    if len(eligible) < profile["min_stocks"]:
        eligible = metrics.nsmallest(profile["min_stocks"], "ann_volatility")

    # Use eligible stocks
    symbols = eligible.index.tolist()
    ret_sub = returns[symbols].dropna(axis=1)
    symbols = ret_sub.columns.tolist()

    mu = ret_sub.mean().values * 252
    cov = ret_sub.cov().values * 252

    weights = mean_variance_optimize(
        mu, cov,
        risk_aversion=profile["risk_aversion"],
        max_weight=profile["max_single_stock"],
        n_stocks=len(symbols),
    )

    portfolio = pd.DataFrame({
        "Symbol": symbols,
        "Weight": weights,
        "Ann_Return": eligible.loc[symbols, "ann_return"].values,
        "Ann_Vol": eligible.loc[symbols, "ann_volatility"].values,
        "Sharpe": eligible.loc[symbols, "sharpe_ratio"].values,
        "Max_DD": eligible.loc[symbols, "max_drawdown"].values,
        "Industry": eligible.loc[symbols, "Industry"].values,
    })
    portfolio = portfolio[portfolio["Weight"] > 0.001].sort_values("Weight", ascending=False)
    portfolio["Weight_pct"] = (portfolio["Weight"] * 100).round(2)

    # Portfolio-level metrics
    w = portfolio["Weight"].values
    port_ret = w @ portfolio["Ann_Return"].values
    port_vol = np.sqrt(w @ (ret_sub[portfolio["Symbol"].tolist()].cov().values * 252) @ w)
    port_sharpe = port_ret / (port_vol + 1e-10)

    portfolio.attrs["portfolio_return"] = round(port_ret, 4)
    portfolio.attrs["portfolio_vol"] = round(port_vol, 4)
    portfolio.attrs["portfolio_sharpe"] = round(port_sharpe, 4)
    portfolio.attrs["profile"] = profile_name

    return portfolio


def build_all_portfolios(df: pd.DataFrame,
                         lookback_start="2019-01-01",
                         lookback_end="2020-12-31"):
    """Build portfolios for all three investor profiles."""
    print("Computing stock metrics...")
    metrics, returns = compute_stock_metrics(df, lookback_start, lookback_end)
    print(f"  {len(metrics)} stocks with valid metrics")

    portfolios = {}
    for name in PROFILES:
        print(f"\nBuilding {name} portfolio...")
        port = build_portfolio(metrics, returns, name)
        portfolios[name] = port
        print(f"  Stocks: {len(port)} | "
              f"Return: {port.attrs['portfolio_return']:.2%} | "
              f"Vol: {port.attrs['portfolio_vol']:.2%} | "
              f"Sharpe: {port.attrs['portfolio_sharpe']:.2f}")

        # Save
        port.to_csv(os.path.join(OUTPUT_DIR, f"portfolio_{name.lower()}.csv"), index=False)

    # Save metrics
    metrics.to_csv(os.path.join(OUTPUT_DIR, "stock_metrics.csv"))
    return portfolios, metrics, returns


if __name__ == "__main__":
    data_path = os.path.join(PROJ_ROOT, "data", "processed", "nifty50_features.parquet")
    df = pd.read_parquet(data_path)
    portfolios, metrics, returns = build_all_portfolios(df)

    for name, port in portfolios.items():
        print(f"\n{'='*50}")
        print(f"{name} Portfolio (Top 10 holdings):")
        print(port[["Symbol", "Weight_pct", "Industry", "Sharpe"]].head(10).to_string(index=False))
