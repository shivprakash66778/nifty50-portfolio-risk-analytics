"""
data_processing.py
──────────────────
Load, clean, and feature-engineer the NIFTY-50 dataset.
Produces a panel DataFrame indexed by (Date, Symbol) with ~60 features.
"""

import os, warnings, glob
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_metadata(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load stock_metadata.csv and return clean DataFrame."""
    meta = pd.read_csv(os.path.join(raw_dir, "stock_metadata.csv"))
    meta.columns = meta.columns.str.strip()
    # Fix M&M symbol mismatch with file name MM.csv
    meta["File_Symbol"] = meta["Symbol"].replace({"M&M": "MM"})
    return meta


def load_individual_csvs(raw_dir: str = RAW_DIR, meta: pd.DataFrame = None) -> pd.DataFrame:
    """Load all individual stock CSVs (not NIFTY50_all) and merge with metadata."""
    if meta is None:
        meta = load_metadata(raw_dir)

    frames = []
    for _, row in meta.iterrows():
        fpath = os.path.join(raw_dir, f"{row['File_Symbol']}.csv")
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        if len(df) <= 1:  # skip INFRATEL (empty)
            continue
        df["Industry"] = row["Industry"]
        df["Company_Name"] = row["Company Name"]
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel.sort_values(["Symbol", "Date"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return panel


def clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, handle missing values."""
    panel = panel.drop_duplicates(subset=["Date", "Symbol"])

    # Fill forward for small gaps per stock, then drop remaining NaN rows
    num_cols = ["Open", "High", "Low", "Close", "VWAP", "Volume", "Turnover"]
    panel[num_cols] = panel.groupby("Symbol")[num_cols].transform(
        lambda x: x.ffill(limit=3)
    )
    panel.dropna(subset=["Close"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return panel


# ── Technical indicators ─────────────────────────────────────────────────────

def _sma(s, w):
    return s.rolling(w, min_periods=w).mean()

def _ema(s, w):
    return s.ewm(span=w, adjust=False).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

def _macd(close):
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    return macd_line, signal, macd_line - signal

def _bollinger(close, w=20, num_std=2):
    sma = _sma(close, w)
    std = close.rolling(w).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pctb = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, pctb


def add_technical_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ~50 technical & statistical features per stock."""
    out_frames = []
    for sym, gdf in panel.groupby("Symbol"):
        g = gdf.copy().sort_values("Date")
        c = g["Close"]
        v = g["Volume"].astype(float)
        h = g["High"].astype(float)
        l = g["Low"].astype(float)
        o = g["Open"].astype(float)

        # ── Returns ──
        for lag in [1, 5, 10, 21, 63]:
            g[f"ret_{lag}d"] = c.pct_change(lag)

        # ── Moving averages ──
        for w in [5, 10, 20, 50, 200]:
            g[f"SMA_{w}"] = _sma(c, w)
            g[f"EMA_{w}"] = _ema(c, w)

        # Price relative to MAs (mean reversion signals)
        for w in [20, 50, 200]:
            g[f"price_to_SMA{w}"] = c / (g[f"SMA_{w}"] + 1e-10) - 1

        # ── RSI ──
        g["RSI_14"] = _rsi(c, 14)

        # ── MACD ──
        macd_l, macd_s, macd_h = _macd(c)
        g["MACD_line"] = macd_l
        g["MACD_signal"] = macd_s
        g["MACD_hist"] = macd_h

        # ── Bollinger Bands ──
        bb_u, bb_l, bb_pct = _bollinger(c, 20)
        g["BB_upper"] = bb_u
        g["BB_lower"] = bb_l
        g["BB_pctB"] = bb_pct

        # ── Volatility ──
        for w in [10, 21, 63]:
            g[f"vol_{w}d"] = c.pct_change().rolling(w).std() * np.sqrt(252)

        # ── Volume features ──
        g["vol_SMA_20"] = _sma(v, 20)
        g["vol_ratio"] = v / (g["vol_SMA_20"] + 1e-10)

        # ── ATR (Average True Range) ──
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)
        g["ATR_14"] = tr.rolling(14).mean()
        g["ATR_pct"] = g["ATR_14"] / (c + 1e-10)

        # ── Momentum ──
        g["momentum_10"] = c / c.shift(10) - 1
        g["momentum_21"] = c / c.shift(21) - 1

        # ── Drawdown from running max ──
        cum_max = c.cummax()
        g["drawdown"] = (c - cum_max) / (cum_max + 1e-10)

        # ── Day-of-week & month ──
        g["dow"] = g["Date"].dt.dayofweek
        g["month"] = g["Date"].dt.month

        # ── Target: forward 21-day return (investment horizon ~1 month) ──
        g["fwd_ret_21d"] = c.shift(-21) / c - 1
        g["fwd_direction"] = (g["fwd_ret_21d"] > 0).astype(int)

        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    return result


def build_dataset(raw_dir: str = RAW_DIR, save: bool = True) -> pd.DataFrame:
    """Full pipeline: load → clean → features → save."""
    print("Loading metadata...")
    meta = load_metadata(raw_dir)
    print(f"  {len(meta)} stocks in metadata")

    print("Loading individual CSVs...")
    panel = load_individual_csvs(raw_dir, meta)
    print(f"  {len(panel)} total rows, {panel['Symbol'].nunique()} stocks")

    print("Cleaning...")
    panel = clean_panel(panel)
    print(f"  {len(panel)} rows after cleaning")

    print("Engineering features...")
    panel = add_technical_features(panel)
    print(f"  {panel.shape[1]} columns after features")

    if save:
        out_path = os.path.join(PROCESSED_DIR, "nifty50_features.parquet")
        panel.to_parquet(out_path, index=False)
        print(f"  Saved to {out_path}")

    return panel


if __name__ == "__main__":
    df = build_dataset()
    print(f"\nFinal shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} – {df['Date'].max()}")
    print(f"Stocks: {df['Symbol'].nunique()}")
    print(f"Columns:\n{list(df.columns)}")
