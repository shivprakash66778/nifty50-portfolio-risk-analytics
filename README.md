# NIFTY-50 Investment Intelligence Platform

> **Cult Open Projects 2026** — Data-Driven Investment Intelligence Using NIFTY-50 Market Data

**Author:** Shiv Prakash Vishwari | IIT Roorkee

An AI-powered investment intelligence platform that transforms 21 years of NIFTY-50 historical market data into actionable investment insights, portfolio recommendations, and risk assessments.

---

## Problem Statement

Develop an intelligent investment platform that goes beyond simple stock-price forecasting to deliver practical investment intelligence. The system must predict future stock behavior, construct risk-profiled portfolios, assess historical risk, provide explainability, detect anomalies, and present results through a deployable application — using only the provided NIFTY-50 dataset. The focus is on creating a decision-support system combining ML, financial analytics, explainability, and user-centric design.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | [NIFTY-50 Stock Market Data (Kaggle)](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data) |
| **Period** | January 3, 2000 – April 30, 2021 |
| **Symbols** | 65 unique (current + historical NIFTY-50 constituents) |
| **Sectors** | 13 industries |
| **Records** | 235,192 (after cleaning) |
| **Columns** | Date, Symbol, Open, High, Low, Close, VWAP, Volume, Turnover, Trades, Deliverable Volume |
| **Metadata** | Company Name, Industry, ISIN Code |

---

## Features Implemented

### Mandatory Tasks

**A. Stock Prediction Engine**
- Regression models (Ridge, XGBoost, LightGBM) predicting 21-day forward returns
- Classification models (Logistic Regression, LightGBM) predicting return direction
- Time-based train/validation/test split (pre-2018 / 2018–mid 2019 / mid 2019–2020)
- 23 engineered features from 6 categories
- Best result: Ridge — RMSE 0.136, 57.7% directional accuracy on test set

**B. Portfolio Construction**
- Markowitz mean-variance optimization with profile-specific constraints
- Three investor profiles (Conservative, Balanced, Aggressive) with calibrated risk aversion, volatility caps, and concentration limits

**C. Risk Assessment**
- 12 metrics per stock: annualized return/volatility, Sharpe, Sortino, Max Drawdown, Calmar, VaR(95%), CVaR(95%), skewness, kurtosis, win rate, beta
- Sector-level risk aggregation and interactive risk-return visualization

### Optional Tasks

**A. Explainable AI** — SHAP TreeExplainer for global feature importance and per-stock local explanations

**B. Personalized Strategies** — Three portfolio profiles with different risk-return tradeoffs and profile-specific constraints

**C. Anomaly Detection** — Four independent detectors: volatility spikes (9,597 events), extreme drawdowns (173,733), volume surges (6,190), single-day crashes (1,163)

**D. Forecasting** — 21-day forward return forecasts for all stocks

**E. Deployment** — 7-page interactive Streamlit dashboard

---

## EDA Summary

- Return distributions exhibit fat tails and negative skewness, confirming the need for robust risk measures beyond variance
- Market volatility clusters around regime shifts: 2008 GFC and 2020 COVID-19 show 3–5x average volatility
- Cross-stock correlations increase during crises, reducing diversification benefits when most needed
- PHARMA and IT sectors provided the best risk-adjusted returns; METALS and ENERGY exhibited highest volatility

---

## Feature Engineering

56 features engineered from raw OHLCV data, 23 selected for modeling:

| Category | Examples | Count |
|----------|----------|-------|
| Returns | 1d, 5d, 10d, 21d, 63d log-returns | 5 |
| Moving Averages | SMA/EMA ratios (price_to_SMA20, 50, 200) | 13 |
| Oscillators | RSI-14, MACD (line, signal, hist), Bollinger %B | 5 |
| Volatility | 10d, 21d, 63d annualized vol, ATR%, volume ratio | 5 |
| Momentum | 10d/21d momentum, drawdown from peak | 3 |
| Calendar | Day of week, month | 2 |

Target variables: `fwd_ret_21d` (21-day forward log-return) and `fwd_direction` (binary up/down).

---

## Model Methodology & Comparison

Time-based split: Train (<2018-01-01, 181,691 rows) / Validation (2018-01 to 2019-06, 18,032 rows) / Test (2019-07 to 2020-12, 18,375 rows).

| Model | Test RMSE | Test R² | Directional Accuracy |
|-------|-----------|---------|---------------------|
| **Ridge** | **0.1360** | **0.0079** | **57.71%** |
| XGBoost | 0.1380 | -0.0213 | 54.36% |
| LightGBM | 0.1382 | -0.0244 | 54.29% |

**Selected Model: Ridge Regression.** Stock-market return prediction is inherently noisy, and low R² values are expected. Directional accuracy of 57.7% is more meaningful for investment decision support than R² alone. The model captures a modest but useful predictive signal and is used as one input into the broader intelligence platform, not as a standalone price predictor.

---

## Portfolio Construction

Mean-variance optimization maximizing: **U = μ − (λ/2) × σ²**

| Profile | λ | Vol Cap | Max Weight | Min Stocks | Stocks | Sharpe |
|---------|---|---------|-----------|-----------|--------|--------|
| Conservative | 5.0 | 15% | 10% | 15 | 10 | 1.61 |
| Balanced | 2.0 | 22% | 12% | 10 | 9 | 1.64 |
| Aggressive | 0.5 | 35% | 15% | 8 | 7 | 1.61 |

**Note:** These portfolios are optimized using historical lookback statistics (2019–2020) and are intended to demonstrate risk-profiled allocation logic rather than guarantee future performance. They should be interpreted as data-driven recommendations that would require further validation before real-world deployment.

---

## Risk Assessment

12 metrics per stock over full history. Key findings:
- **Best Sharpe:** BHARTIARTL (1.18), UTIBANK (0.95), SHREECEM (0.83)
- **Highest beta:** AXISBANK (1.46), TATAMOTORS (1.35)
- Fat tails prevalent across the universe

---

## Explainability (SHAP)

Top return prediction drivers (mean |SHAP|):
1. **month** (0.0054) — calendar seasonality in Indian markets
2. **ATR_pct** (0.0047) — volatility regime indicator
3. **ret_63d** (0.0044) — medium-term momentum
4. **price_to_SMA200** (0.0033) — long-term trend position

Per-stock local explanations show which factors push each prediction up or down.

---

## Anomaly Detection

| Detector | Threshold | Events |
|----------|-----------|--------|
| Volatility Spike | 21d vol Z-score > 2.5 | 9,597 |
| Extreme Drawdown | Drawdown < -15% | 173,733 |
| Volume Surge | Volume > 3x 50-day avg | 6,190 |
| Single-Day Crash | Daily return < -8% | 1,163 |

2008 (GFC) and 2020 (COVID-19) produced the highest anomaly density. Volatility spikes tend to precede crash events by several trading days.

---

## Dashboard

7-page interactive Streamlit application:
1. **Market Overview** — sector distribution, equal-weighted index proxy, sector heatmap, correlation matrix
2. **Stock Analysis** — candlestick charts, SMA overlays, RSI, MACD, volume, model predictions
3. **Portfolio Builder** — 3 profiles with allocation pie charts, sector breakdown, holdings table, profile comparison
4. **Risk Assessment** — risk-return scatter, top/bottom tables, VaR distribution
5. **Anomaly Detection** — anomaly summary, timeline by year/type, per-stock breakdown
6. **Explainability** — global SHAP importance, per-stock SHAP waterfall, feature importance comparison
7. **Model Performance** — regression/classification metrics, visual comparison, interpretation notes

---

## Results Summary

| Module | Key Metric | Value |
|--------|-----------|-------|
| Prediction | Best Directional Accuracy | 57.71% |
| Prediction | Best RMSE | 0.136 |
| Portfolio | Sharpe (all profiles) | >1.6 |
| Risk | Metrics per stock | 12 |
| Explainability | SHAP features | 23 |
| Anomaly | Detectors | 4 |
| Dashboard | Interactive pages | 7 |

---

## Environment Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<username>/nifty50-investment-intelligence.git
cd nifty50-investment-intelligence
pip install -r requirements.txt
```

### Dataset Setup
1. Download from [Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
2. Extract all CSV files into `data/raw/`

---

## Running the Application

### Quick Start (Dashboard)
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501`.

### Reproducing Results (Full Pipeline)

```bash
# Step 1: Process raw data → engineered features
python src/data_processing.py

# Step 2: Train prediction models
python src/prediction_engine.py

# Step 3: Construct portfolios
python src/portfolio.py

# Step 4: Compute risk metrics
python src/risk_assessment.py

# Step 5: Run anomaly detection
python src/anomaly_detection.py

# Step 6: Generate SHAP explanations
python src/explainability.py

# Step 7: Generate report charts
python src/generate_charts.py

# Step 8: Generate technical report PDF
python reports/generate_report.py

# Step 9: Launch dashboard
streamlit run app/app.py
```

---

## Repository Structure

```
nifty50-investment-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                               # Source modules
│   ├── __init__.py
│   ├── data_processing.py             # Data loading, cleaning, feature engineering
│   ├── prediction_engine.py           # ML models (Ridge, XGBoost, LightGBM)
│   ├── portfolio.py                   # Mean-variance portfolio optimization
│   ├── risk_assessment.py             # 12 risk metrics per stock
│   ├── anomaly_detection.py           # 4 anomaly detectors
│   ├── explainability.py              # SHAP-based explanations
│   └── generate_charts.py             # EDA & report visualizations
│
├── app/
│   └── app.py                         # Streamlit dashboard (7 pages)
│
├── models/                            # Saved model files (.pkl)
│
├── data/
│   ├── raw/                           # Original CSV files (52 files)
│   └── processed/
│       └── nifty50_features.parquet   # Engineered features (235K rows × 56 cols)
│
├── outputs/
│   ├── charts/                        # Generated visualizations (10 PNG files)
│   ├── model_results.json
│   ├── feature_importance.csv
│   ├── shap_importance.csv
│   ├── risk_report.csv
│   ├── stock_metrics.csv
│   ├── anomalies.csv
│   ├── anomaly_summary.csv
│   ├── portfolio_conservative.csv
│   ├── portfolio_balanced.csv
│   └── portfolio_aggressive.csv
│
├── notebooks/
│   └── analysis.ipynb                 # Documented analysis notebook
│
├── reports/
│   ├── Technical_Report.pdf           # Technical report
│   └── generate_report.py            # Report generation script
│
└── config/                            # Reserved
```

---

## Limitations

1. **Survivorship bias** — dataset includes only current/recent NIFTY-50 constituents
2. **No transaction costs** — portfolio returns don't account for trading frictions
3. **Fixed prediction horizon** — 21-day forward returns; adaptive horizons could improve results
4. **Static risk-free rate** — 6% assumed; actual rates vary over the 21-year period
5. **No fundamental data** — constrained to price/volume features per competition rules
6. **COVID regime shift** — model performance degrades during unprecedented market conditions

---

## Future Improvements

- Ensemble Ridge + LightGBM predictions for improved robustness
- Black-Litterman portfolio optimization incorporating model views
- Hidden Markov Model regime detection for dynamic model switching
- Multi-horizon predictions (5d, 10d, 63d) with horizon-specific models
- Transaction cost modeling in portfolio optimization

---

---

## Final Submission Checklist

- [x] Working prototype (Streamlit dashboard) included
- [x] Technical report under 12 pages
- [x] README with setup and reproduction instructions
- [x] Source code included (`src/` with 7 modules)
- [x] Model files included (`models/` with 5 .pkl files)
- [x] Processed outputs included (`outputs/`)
- [x] Streamlit dashboard included (`app/app.py`, 7 pages)
- [x] Risk assessment included (12 metrics per stock)
- [x] Explainability included (SHAP global + per-stock)
- [x] Anomaly detection included (4 independent detectors)
- [x] Analysis notebook included (`notebooks/analysis.ipynb`)
- [ ] Hosted Streamlit link — add if deployed (e.g., Streamlit Community Cloud)


---

*Built with Python, scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, and Plotly.*
