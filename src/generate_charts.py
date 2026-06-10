"""
generate_charts.py
──────────────────
Generate all EDA and result visualizations for the technical report.
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def main():
    df = pd.read_parquet(os.path.join(PROJ_ROOT, "data", "processed", "nifty50_features.parquet"))

    # 1. Price evolution — top 10 stocks by market cap proxy (volume * close)
    print("Generating price evolution chart...")
    recent = df[df["Date"] >= "2010-01-01"]
    avg_mcap = recent.groupby("Symbol").apply(lambda g: (g["Close"] * g["Volume"]).mean())
    top10 = avg_mcap.nlargest(10).index
    fig, ax = plt.subplots(figsize=(12, 6))
    for sym in top10:
        s = df[df["Symbol"] == sym].set_index("Date")["Close"]
        normed = s / s.iloc[0]
        ax.plot(normed.index, normed.values, label=sym, linewidth=1)
    ax.set_title("Normalized Price Evolution — Top 10 NIFTY-50 Stocks", fontsize=14)
    ax.set_ylabel("Normalized Price (base=1)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "price_evolution.png"), dpi=150)
    plt.close()

    # 2. Return distribution
    print("Generating return distribution...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (col, title) in enumerate([
        ("ret_1d", "Daily Returns"), ("ret_21d", "21-Day Returns"), ("ret_63d", "63-Day Returns")
    ]):
        data = df[col].dropna()
        axes[i].hist(data.clip(-0.3, 0.3), bins=100, density=True, alpha=0.7, color="steelblue")
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel("Return")
        axes[i].axvline(0, color="red", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "return_distributions.png"), dpi=150)
    plt.close()

    # 3. Volatility over time (market-wide)
    print("Generating volatility chart...")
    mkt_vol = df.groupby("Date")["vol_21d"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(mkt_vol.index, mkt_vol.values, linewidth=0.8, color="crimson")
    ax.fill_between(mkt_vol.index, mkt_vol.values, alpha=0.15, color="crimson")
    ax.set_title("Market-Wide Average 21-Day Annualized Volatility", fontsize=14)
    ax.set_ylabel("Volatility")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "market_volatility.png"), dpi=150)
    plt.close()

    # 4. Sector returns boxplot
    print("Generating sector boxplot...")
    sector_rets = df[["Industry", "ret_21d"]].dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sector_rets.groupby("Industry")["ret_21d"].median().sort_values(ascending=False).index
    sns.boxplot(data=sector_rets, x="Industry", y="ret_21d", order=order,
                ax=ax, showfliers=False, palette="Set3")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("21-Day Return Distribution by Sector", fontsize=14)
    ax.set_ylabel("21-Day Return")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "sector_boxplot.png"), dpi=150)
    plt.close()

    # 5. Correlation heatmap (features)
    print("Generating feature correlation heatmap...")
    feat_cols = ["ret_1d", "ret_5d", "ret_21d", "RSI_14", "MACD_hist", "BB_pctB",
                 "vol_21d", "vol_ratio", "ATR_pct", "momentum_21", "drawdown"]
    corr = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "feature_correlation.png"), dpi=150)
    plt.close()

    # 6. Model comparison bar chart
    print("Generating model comparison chart...")
    import json
    results_path = os.path.join(OUTPUT_DIR, "model_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        reg = results["regression"]
        models = list(reg.keys())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        metrics = [("test_RMSE", "Test RMSE (lower=better)", "salmon"),
                   ("test_R2", "Test R² (higher=better)", "skyblue"),
                   ("test_DirAcc", "Test Dir. Accuracy", "lightgreen")]
        for i, (m, title, color) in enumerate(metrics):
            vals = [reg[model][m] for model in models]
            axes[i].bar(models, vals, color=color, edgecolor="black", linewidth=0.5)
            axes[i].set_title(title, fontsize=11)
            for j, v in enumerate(vals):
                axes[i].text(j, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, "model_comparison.png"), dpi=150)
        plt.close()

    # 7. Risk-return scatter
    print("Generating risk-return scatter...")
    risk_path = os.path.join(OUTPUT_DIR, "risk_report.csv")
    if os.path.exists(risk_path):
        risk_df = pd.read_csv(risk_path)
        risk_df = risk_df.dropna(subset=["Ann_Return", "Ann_Volatility"])
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(risk_df["Ann_Volatility"], risk_df["Ann_Return"],
                            c=risk_df["Sharpe_Ratio"], cmap="RdYlGn",
                            s=60, edgecolor="black", linewidth=0.3, alpha=0.8)
        for _, row in risk_df.iterrows():
            ax.annotate(row["Name"], (row["Ann_Volatility"], row["Ann_Return"]),
                       fontsize=6, alpha=0.7)
        plt.colorbar(scatter, label="Sharpe Ratio")
        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Return")
        ax.set_title("Risk-Return Profile of NIFTY-50 Stocks", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, "risk_return_scatter.png"), dpi=150)
        plt.close()

    # 8. Portfolio allocation charts
    print("Generating portfolio charts...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, profile in enumerate(["conservative", "balanced", "aggressive"]):
        port_path = os.path.join(OUTPUT_DIR, f"portfolio_{profile}.csv")
        if os.path.exists(port_path):
            port = pd.read_csv(port_path)
            axes[i].pie(port["Weight_pct"], labels=port["Symbol"],
                       autopct="%1.0f%%", textprops={"fontsize": 7})
            axes[i].set_title(f"{profile.title()} Portfolio", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "portfolio_allocations.png"), dpi=150)
    plt.close()

    print(f"\nAll charts saved to {CHART_DIR}")
    print(f"Files: {os.listdir(CHART_DIR)}")


if __name__ == "__main__":
    main()
