"""
explainability.py
─────────────────
SHAP-based explanations for stock predictions and portfolio decisions.
"""

import os, warnings, joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(PROJ_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    "price_to_SMA20", "price_to_SMA50", "price_to_SMA200",
    "RSI_14", "MACD_line", "MACD_signal", "MACD_hist",
    "BB_pctB",
    "vol_10d", "vol_21d", "vol_63d",
    "vol_ratio", "ATR_pct",
    "momentum_10", "momentum_21",
    "drawdown", "dow", "month",
]


def compute_shap_values(model, X_sample: pd.DataFrame, model_type="tree"):
    """Compute SHAP values for a sample."""
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def generate_shap_plots(df: pd.DataFrame):
    """Generate and save SHAP summary plots."""
    model_path = os.path.join(MODEL_DIR, "lgb_reg.pkl")
    if not os.path.exists(model_path):
        print("No LightGBM model found. Run prediction_engine.py first.")
        return

    model = joblib.load(model_path)
    # Use test period data
    test_data = df[(df["Date"] >= "2019-07-01") & (df["Date"] < "2021-01-01")]
    test_data = test_data.dropna(subset=FEATURE_COLS)
    X_sample = test_data[FEATURE_COLS].sample(min(2000, len(test_data)), random_state=42)

    print("Computing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X_sample)

    # Summary plot (bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "shap_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_importance.png")

    # Summary plot (dot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_summary.png")

    # Mean absolute SHAP importance
    mean_shap = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    mean_shap.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)

    return shap_values, X_sample


def explain_single_prediction(df: pd.DataFrame, symbol: str, date: str = None):
    """Generate waterfall explanation for a single stock on a given date."""
    model = joblib.load(os.path.join(MODEL_DIR, "lgb_reg.pkl"))
    sub = df[df["Symbol"] == symbol].dropna(subset=FEATURE_COLS).sort_values("Date")

    if date:
        row = sub[sub["Date"] == date]
    else:
        row = sub.tail(1)

    if len(row) == 0:
        print(f"No data for {symbol} on {date}")
        return None

    X = row[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    explanation_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "value": X.values[0],
        "shap_value": sv[0]
    }).sort_values("shap_value", key=abs, ascending=False)

    prediction = model.predict(X)[0]
    return prediction, explanation_df


if __name__ == "__main__":
    data_path = os.path.join(PROJ_ROOT, "data", "processed", "nifty50_features.parquet")
    df = pd.read_parquet(data_path)
    generate_shap_plots(df)

    # Example single explanation
    pred, exp = explain_single_prediction(df, "RELIANCE", "2020-06-01")
    if pred is not None:
        print(f"\nRELIANCE prediction on 2020-06-01: {pred:.4f}")
        print("Top factors:")
        print(exp.head(5).to_string(index=False))
