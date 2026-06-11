"""
anomaly_detection.py
────────────────────
Detect market anomalies: volatility spikes, extreme drawdowns, volume surges.
Uses statistical Z-score thresholding and rolling percentile methods.
"""

import os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs")


def detect_volatility_spikes(df: pd.DataFrame, window: int = 21,
                              z_thresh: float = 2.5) -> pd.DataFrame:
    """Detect days where realized volatility exceeds historical norms."""
    anomalies = []
    for sym, g in df.groupby("Symbol"):
        g = g.sort_values("Date").copy()
        ret = g["Close"].pct_change()
        rolling_vol = ret.rolling(window).std() * np.sqrt(252)
        vol_mean = rolling_vol.rolling(252).mean()
        vol_std = rolling_vol.rolling(252).std()
        z_score = (rolling_vol - vol_mean) / (vol_std + 1e-10)

        spikes = g[z_score > z_thresh][["Date", "Symbol", "Close"]].copy()
        spikes["vol_zscore"] = z_score[z_score > z_thresh].values
        spikes["anomaly_type"] = "volatility_spike"
        anomalies.append(spikes)

    return pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()


def detect_extreme_drawdowns(df: pd.DataFrame, threshold: float = -0.15) -> pd.DataFrame:
    """Detect drawdown events exceeding threshold."""
    anomalies = []
    for sym, g in df.groupby("Symbol"):
        g = g.sort_values("Date").copy()
        prices = g.set_index("Date")["Close"]
        cum_max = prices.cummax()
        dd = (prices - cum_max) / cum_max

        extreme = dd[dd < threshold]
        if len(extreme) > 0:
            for date, val in extreme.items():
                anomalies.append({
                    "Date": date, "Symbol": sym,
                    "drawdown": round(val, 4),
                    "anomaly_type": "extreme_drawdown"
                })

    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()


def detect_volume_anomalies(df: pd.DataFrame, window: int = 50,
                             multiplier: float = 3.0) -> pd.DataFrame:
    """Detect unusual trading volume."""
    anomalies = []
    for sym, g in df.groupby("Symbol"):
        g = g.sort_values("Date").copy()
        vol = g["Volume"].astype(float)
        vol_ma = vol.rolling(window).mean()
        ratio = vol / (vol_ma + 1e-10)

        surges = g[ratio > multiplier][["Date", "Symbol", "Close", "Volume"]].copy()
        surges["vol_ratio"] = ratio[ratio > multiplier].values
        surges["anomaly_type"] = "volume_surge"
        anomalies.append(surges)

    return pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()


def detect_single_day_crashes(df: pd.DataFrame, threshold: float = -0.08) -> pd.DataFrame:
    """Detect single-day crashes (return < threshold)."""
    anomalies = []
    for sym, g in df.groupby("Symbol"):
        g = g.sort_values("Date").copy()
        ret = g["Close"].pct_change()
        crashes = g[ret < threshold][["Date", "Symbol", "Close"]].copy()
        crashes["daily_return"] = ret[ret < threshold].values
        crashes["anomaly_type"] = "single_day_crash"
        anomalies.append(crashes)

    return pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Run all anomaly detectors and return combined results."""
    print("Detecting volatility spikes...")
    vol_spikes = detect_volatility_spikes(df)
    print(f"  {len(vol_spikes)} volatility spike events")

    print("Detecting extreme drawdowns...")
    drawdowns = detect_extreme_drawdowns(df)
    print(f"  {len(drawdowns)} extreme drawdown events")

    print("Detecting volume anomalies...")
    vol_anom = detect_volume_anomalies(df)
    print(f"  {len(vol_anom)} volume surge events")

    print("Detecting single-day crashes...")
    crashes = detect_single_day_crashes(df)
    print(f"  {len(crashes)} crash events")

    all_anomalies = pd.concat([vol_spikes, drawdowns, vol_anom, crashes],
                               ignore_index=True)
    all_anomalies.to_csv(os.path.join(OUTPUT_DIR, "anomalies.csv"), index=False)
    print(f"\nTotal anomalies saved: {len(all_anomalies)}")

    # Summary by year
    if len(all_anomalies) > 0 and "Date" in all_anomalies.columns:
        all_anomalies["Date"] = pd.to_datetime(all_anomalies["Date"])
        summary = all_anomalies.groupby(
            [all_anomalies["Date"].dt.year, "anomaly_type"]
        ).size().unstack(fill_value=0)
        summary.to_csv(os.path.join(OUTPUT_DIR, "anomaly_summary.csv"))

    return all_anomalies


if __name__ == "__main__":
    data_path = os.path.join(PROJ_ROOT, "data", "processed", "nifty50_features.parquet")
    df = pd.read_parquet(data_path)
    anomalies = run_anomaly_detection(df)
