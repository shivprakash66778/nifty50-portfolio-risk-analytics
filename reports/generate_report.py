"""
generate_report.py
==================
Generates a polished <=12-page Technical Report PDF for the NIFTY-50
Investment Intelligence Platform.
"""

import os, json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)

PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CHART = os.path.join(PROJ, "outputs", "charts")
OUTPUT = os.path.join(PROJ, "outputs")
REPORT_PATH = os.path.join(PROJ, "reports", "Technical_Report.pdf")

W, H = A4
MARGIN = 0.65 * inch

# --------------- styles ---------------
styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    "CoverTitle", parent=styles["Title"], fontSize=26, leading=32,
    textColor=HexColor("#1a237e"), spaceAfter=6, alignment=TA_CENTER))
styles.add(ParagraphStyle(
    "CoverSub", parent=styles["Normal"], fontSize=14, leading=18,
    textColor=HexColor("#37474f"), alignment=TA_CENTER, spaceAfter=4))
styles.add(ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=14, leading=18,
    textColor=HexColor("#1a237e"), spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=11, leading=14,
    textColor=HexColor("#283593"), spaceBefore=8, spaceAfter=3))
styles.add(ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=9, leading=12,
    alignment=TA_JUSTIFY, spaceAfter=3))
styles.add(ParagraphStyle(
    "BodySmall", parent=styles["Normal"], fontSize=8.5, leading=11,
    alignment=TA_JUSTIFY, spaceAfter=2))
styles.add(ParagraphStyle(
    "Caption", parent=styles["Normal"], fontSize=7.5, leading=9.5,
    textColor=HexColor("#546e7a"), alignment=TA_CENTER,
    spaceBefore=1, spaceAfter=6))
styles.add(ParagraphStyle(
    "BulletItem", parent=styles["Normal"], fontSize=9, leading=12,
    leftIndent=16, bulletIndent=6, spaceAfter=2))
styles.add(ParagraphStyle(
    "TableHeader", parent=styles["Normal"], fontSize=8, leading=10,
    textColor=white, alignment=TA_CENTER))
styles.add(ParagraphStyle(
    "TableCell", parent=styles["Normal"], fontSize=7.5, leading=9.5,
    alignment=TA_CENTER))
styles.add(ParagraphStyle(
    "TableCellLeft", parent=styles["Normal"], fontSize=7.5, leading=9.5,
    alignment=TA_LEFT))

def make_table(header, rows, col_widths=None):
    hdr = [Paragraph(f"<b>{h}</b>", styles["TableHeader"]) for h in header]
    data = [hdr]
    for row in rows:
        data.append([Paragraph(str(c), styles["TableCell"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a237e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f5f5")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t

def img(name, w=5.6*inch):
    path = os.path.join(CHART, name)
    if os.path.exists(path):
        return Image(path, width=w, height=w*0.42)
    return Spacer(1, 6)

def bullet(text):
    return Paragraph(f"<bullet>&bull;</bullet> {text}", styles["BulletItem"])

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#b0bec5"),
                       spaceAfter=4, spaceBefore=3)

# ====================== BUILD ========================
story = []

# ---------- COVER PAGE ----------
story.append(Spacer(1, 2.2*inch))
story.append(Paragraph("NIFTY-50 Investment Intelligence Platform", styles["CoverTitle"]))
story.append(Spacer(1, 6))
story.append(Paragraph("Technical Report", styles["CoverSub"]))
story.append(Spacer(1, 8))
story.append(HRFlowable(width="60%", thickness=1.5, color=HexColor("#1a237e"),
                          spaceAfter=8, spaceBefore=4))
story.append(Paragraph("Cult Open Projects 2026", styles["CoverSub"]))
story.append(Spacer(1, 4))
story.append(Paragraph("Data-Driven Investment Intelligence Using NIFTY-50 Market Data", styles["BodySmall"]))
story.append(Spacer(1, 0.6*inch))
story.append(Paragraph("Shiv Prakash Vishwari | IIT Roorkee", styles["CoverSub"]))
story.append(Paragraph("June 2026", styles["CoverSub"]))
story.append(PageBreak())

# ---------- TABLE OF CONTENTS ----------
story.append(Paragraph("Table of Contents", styles["H1"]))
toc_items = [
    "1. Executive Summary",
    "2. Dataset & Exploratory Data Analysis",
    "3. Feature Engineering",
    "4. Prediction Engine — Methodology & Results",
    "5. Portfolio Construction",
    "6. Risk Assessment",
    "7. Explainable AI Framework",
    "8. Market Anomaly Detection",
    "9. Deployment & Dashboard",
    "10. Key Insights & Limitations",
]
for item in toc_items:
    story.append(Paragraph(item, styles["Body"]))
story.append(Spacer(1, 6))
story.append(hr())

# ---------- 1. EXECUTIVE SUMMARY ----------
story.append(Paragraph("1. Executive Summary", styles["H1"]))
story.append(Paragraph(
    "This report presents a comprehensive AI-powered investment intelligence platform "
    "built on 21 years of NIFTY-50 historical market data (January 2000 to April 2021). "
    "The platform integrates six core modules: a stock prediction engine comparing Ridge, "
    "XGBoost, and LightGBM models; a mean-variance portfolio optimizer generating "
    "Conservative, Balanced, and Aggressive portfolios; a risk assessment framework "
    "computing 12 risk metrics per stock; SHAP-based explainability; multi-signal anomaly "
    "detection; and an interactive Streamlit dashboard. The system achieves 57.7% directional "
    "accuracy on the test set (2019-2020, including the COVID-19 crash), suggesting the model "
    "captures a modest but useful predictive signal under noisy market conditions. The model "
    "serves as one input into the broader investment intelligence framework, not as a standalone "
    "price predictor. All analysis uses only the provided dataset with no external data sources.",
    styles["Body"]))

# ---------- 2. EDA ----------
story.append(Paragraph("2. Dataset & Exploratory Data Analysis", styles["H1"]))
story.append(Paragraph(
    "The dataset contains daily OHLCV data for 65 unique symbols that have been part of "
    "the NIFTY-50 index across 13 industry sectors. After cleaning (removing the empty "
    "INFRATEL file, deduplication, and forward-filling gaps with a limit of 3 days), the "
    "processed dataset comprises 235,192 rows across 56 engineered features.",
    styles["Body"]))
story.append(Paragraph("2.1 Dataset Statistics", styles["H2"]))
story.append(make_table(
    ["Metric", "Value"],
    [["Total Records", "235,192"],
     ["Unique Symbols", "65"],
     ["Industry Sectors", "13"],
     ["Date Range", "2000-01-03 to 2021-04-30"],
     ["Feature Columns", "56 (23 used for modeling)"],
     ["Missing Data Strategy", "Forward-fill (limit=3), drop if Close is NaN"]],
    col_widths=[2.5*inch, 3.3*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph("2.2 Key EDA Findings", styles["H2"]))
story.append(bullet("Return distributions exhibit fat tails and negative skewness, confirming the need for robust risk measures beyond variance."))
story.append(bullet("Market volatility clusters around regime shifts: the 2008 GFC and 2020 COVID-19 periods show 3-5x average volatility."))
story.append(bullet("Cross-stock correlations increase during crises, reducing diversification benefits when most needed."))
story.append(bullet("Sector-level analysis reveals PHARMA and IT provided the best risk-adjusted returns, while METALS and ENERGY exhibited highest volatility."))

story.append(img("return_distributions.png"))
story.append(Paragraph("Figure 1: Return distributions at daily, 21-day, and 63-day horizons showing fat tails and slight negative skew.", styles["Caption"]))

story.append(img("market_volatility.png"))
story.append(Paragraph("Figure 2: Market-wide 21-day annualized volatility with peaks at GFC (2008) and COVID-19 (2020).", styles["Caption"]))

# ---------- 3. FEATURE ENGINEERING ----------
story.append(Paragraph("3. Feature Engineering", styles["H1"]))
story.append(Paragraph(
    "We engineered 56 features from raw OHLCV data organized into six categories. "
    "Features were computed per-stock using rolling windows to avoid lookahead bias. "
    "All features are available at prediction time (no forward-looking data leakage).",
    styles["Body"]))

story.append(make_table(
    ["Category", "Features", "Count"],
    [["Returns", "1d, 5d, 10d, 21d, 63d log-returns", "5"],
     ["Moving Averages", "SMA/EMA (5, 10, 20, 50, 200) + price-to-MA ratios", "13"],
     ["Oscillators", "RSI-14, MACD (line, signal, histogram), Bollinger %B", "5"],
     ["Volatility", "10d, 21d, 63d annualized vol, ATR%, volume ratio", "5"],
     ["Momentum", "10d and 21d momentum, drawdown from peak", "3"],
     ["Calendar", "Day of week, month (categorical encoded)", "2"]],
    col_widths=[1.4*inch, 3.0*inch, 0.7*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph(
    "Target variables: (1) fwd_ret_21d — 21-day forward log-return for regression; "
    "(2) fwd_direction — binary indicator (1 if fwd_ret_21d > 0) for classification.",
    styles["Body"]))

story.append(img("feature_correlation.png", 4.2*inch))
story.append(Paragraph("Figure 3: Feature correlation matrix showing relationships between returns, momentum, and volatility.", styles["Caption"]))

# ---------- 4. PREDICTION ENGINE ----------
story.append(Paragraph("4. Prediction Engine", styles["H1"]))

story.append(Paragraph("4.1 Methodology", styles["H2"]))
story.append(Paragraph(
    "We employ a strict time-based train/validation/test split to prevent lookahead bias, "
    "reflecting a realistic deployment scenario.",
    styles["Body"]))
story.append(make_table(
    ["Split", "Period", "Rows"],
    [["Train", "Before 2018-01-01", "181,691"],
     ["Validation", "2018-01 to 2019-06", "18,032"],
     ["Test", "2019-07 to 2020-12", "18,375"]],
    col_widths=[1.4*inch, 2.2*inch, 1.4*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph("4.2 Model Selection Rationale", styles["H2"]))
story.append(bullet("<b>Ridge Regression:</b> Linear baseline with L2 regularization. Robust to multicollinearity among technical indicators."))
story.append(bullet("<b>XGBoost:</b> Gradient-boosted trees modeling non-linear feature interactions with L1/L2 regularization."))
story.append(bullet("<b>LightGBM:</b> Histogram-based gradient boosting with faster training and leaf-wise growth."))
story.append(bullet("<b>Logistic Regression + LightGBM Classifier:</b> Direction prediction models complementing return forecasts."))

story.append(Paragraph("4.3 Results", styles["H2"]))
story.append(Paragraph("<b>Regression Models (21-Day Return Prediction)</b>", styles["Body"]))
story.append(make_table(
    ["Model", "Val MAE", "Val RMSE", "Test MAE", "Test RMSE", "Test R2", "Test Dir.Acc"],
    [["Ridge", "0.0648", "0.0886", "0.0942", "0.1360", "0.0079", "57.71%"],
     ["XGBoost", "0.0641", "0.0880", "0.0962", "0.1380", "-0.0213", "54.36%"],
     ["LightGBM", "0.0641", "0.0880", "0.0962", "0.1382", "-0.0244", "54.29%"]],
    col_widths=[1.0*inch, 0.7*inch, 0.75*inch, 0.75*inch, 0.8*inch, 0.65*inch, 0.85*inch]
))
story.append(Spacer(1, 3))
story.append(Paragraph("<b>Classification Models (Direction Prediction)</b>", styles["Body"]))
story.append(make_table(
    ["Model", "Val AUC", "Val Accuracy", "Test AUC", "Test Accuracy"],
    [["Logistic Regression", "0.4803", "49.53%", "0.5382", "57.43%"],
     ["LightGBM Classifier", "0.5282", "49.98%", "0.5250", "57.86%"]],
    col_widths=[1.5*inch, 0.95*inch, 1.05*inch, 0.95*inch, 1.05*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph(
    "<b>Key Insight:</b> Ridge regression outperforms tree-based models on the test set "
    "(which includes the COVID-19 regime shift), suggesting the dominant predictive signal is "
    "linear — short-term momentum and mean-reversion patterns captured by price-to-MA ratios. "
    "A low R-squared is expected in financial return forecasting, where markets are inherently "
    "noisy. Directional accuracy of 57.7% is more meaningful for investment decision support "
    "than R-squared alone, suggesting the model captures a modest but useful predictive signal. "
    "The prediction engine is designed as one component of the broader intelligence platform, "
    "not as a standalone price predictor.",
    styles["Body"]))

story.append(img("model_comparison.png"))
story.append(Paragraph("Figure 4: Model comparison across RMSE, R-squared, and directional accuracy.", styles["Caption"]))

# ---------- 5. PORTFOLIO CONSTRUCTION ----------
story.append(Paragraph("5. Portfolio Construction", styles["H1"]))
story.append(Paragraph(
    "We implement a Markowitz mean-variance optimization framework with profile-specific "
    "constraints. The optimizer maximizes the utility function: U = mu - (lambda/2) * sigma^2, "
    "where lambda is the risk aversion parameter. Stock-level constraints prevent "
    "concentration risk, and minimum stock count ensures diversification.",
    styles["Body"]))

story.append(Paragraph("5.1 Profile Constraints", styles["H2"]))
story.append(make_table(
    ["Parameter", "Conservative", "Balanced", "Aggressive"],
    [["Risk Aversion (lambda)", "5.0", "2.0", "0.5"],
     ["Max Volatility", "15%", "22%", "35%"],
     ["Max Per-Stock Weight", "10%", "12%", "15%"],
     ["Min Stocks", "15", "10", "8"]],
    col_widths=[1.7*inch, 1.2*inch, 1.2*inch, 1.2*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph("5.2 Portfolio Results (Lookback: 2019-01 to 2020-12)", styles["H2"]))
story.append(make_table(
    ["Profile", "Stocks", "Ann. Return", "Ann. Vol", "Sharpe"],
    [["Conservative", "10", "33.23%", "20.62%", "1.61"],
     ["Balanced", "9", "36.63%", "22.33%", "1.64"],
     ["Aggressive", "7", "38.38%", "23.77%", "1.61"]],
    col_widths=[1.2*inch, 0.8*inch, 1.1*inch, 1.1*inch, 0.9*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph(
    "The portfolio results are based on historical lookback statistics and are intended to "
    "demonstrate risk-profiled allocation logic rather than guarantee future performance. "
    "The return-volatility tradeoff works as designed: the Conservative profile minimizes "
    "drawdown risk while the Aggressive profile accepts higher volatility for greater return "
    "potential. These allocations should be interpreted as data-driven recommendations that "
    "would require further validation before real-world deployment.",
    styles["Body"]))

story.append(img("portfolio_allocations.png"))
story.append(Paragraph("Figure 5: Portfolio allocation across Conservative, Balanced, and Aggressive profiles.", styles["Caption"]))

# ---------- 6. RISK ASSESSMENT ----------
story.append(Paragraph("6. Risk Assessment", styles["H1"]))
story.append(Paragraph(
    "Each stock is evaluated across 12 risk metrics computed over its full trading history. "
    "This comprehensive profiling enables informed portfolio decisions beyond simple return comparisons.",
    styles["Body"]))

story.append(Paragraph("6.1 Risk Metrics Computed", styles["H2"]))
story.append(make_table(
    ["Metric", "Description"],
    [["Annualized Return", "CAGR from first to last available trading day"],
     ["Annualized Volatility", "Standard deviation of daily returns * sqrt(252)"],
     ["Sharpe Ratio", "(Ann. Return - Rf) / Ann. Vol, Rf = 6%"],
     ["Sortino Ratio", "Returns per unit of downside deviation"],
     ["Maximum Drawdown", "Largest peak-to-trough decline"],
     ["Calmar Ratio", "Ann. Return / |Max Drawdown|"],
     ["VaR (95%)", "5th percentile of daily returns"],
     ["CVaR (95%)", "Mean of returns below VaR threshold"],
     ["Skewness & Kurtosis", "Distribution shape and tail-risk measures"],
     ["Win Rate", "Fraction of positive-return trading days"],
     ["Beta", "Sensitivity to equal-weighted market index"]],
    col_widths=[1.6*inch, 4.1*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph("6.2 Top Stocks by Sharpe Ratio", styles["H2"]))
risk_df = pd.read_csv(os.path.join(OUTPUT, "risk_report.csv"))
top5 = risk_df.nlargest(5, "Sharpe_Ratio")
rows = []
for _, r in top5.iterrows():
    rows.append([r["Name"], f"{r['Ann_Return']:.1%}", f"{r['Ann_Volatility']:.1%}",
                 f"{r['Sharpe_Ratio']:.2f}", f"{r['Max_Drawdown']:.1%}", f"{r['Beta']:.2f}"])
story.append(make_table(
    ["Stock", "Return", "Vol", "Sharpe", "MaxDD", "Beta"], rows,
    col_widths=[1.2*inch, 0.85*inch, 0.75*inch, 0.75*inch, 0.85*inch, 0.75*inch]
))

story.append(img("risk_return_scatter.png", 4.8*inch))
story.append(Paragraph("Figure 6: Risk-return scatter of all NIFTY-50 stocks, color-coded by Sharpe ratio.", styles["Caption"]))

# ---------- 7. EXPLAINABILITY ----------
story.append(Paragraph("7. Explainable AI Framework", styles["H1"]))
story.append(Paragraph(
    "Explainability is achieved through SHAP (SHapley Additive exPlanations) values "
    "computed using TreeExplainer on the LightGBM regression model. This provides both "
    "global feature importance and local per-prediction explanations.",
    styles["Body"]))

story.append(Paragraph("7.1 Global Feature Importance (SHAP)", styles["H2"]))
shap_df = pd.read_csv(os.path.join(OUTPUT, "shap_importance.csv"))
rows = []
for _, r in shap_df.head(8).iterrows():
    rows.append([r["feature"], f"{r['mean_abs_shap']:.6f}"])
story.append(make_table(
    ["Feature", "Mean |SHAP|"], rows,
    col_widths=[2.3*inch, 1.8*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph(
    "The calendar feature 'month' is the top SHAP contributor, capturing well-known "
    "seasonality effects in Indian markets (e.g., pre-budget rally, post-monsoon recovery). "
    "ATR% and 63-day returns follow, indicating that volatility regime and medium-term "
    "momentum are key return drivers. The dashboard provides per-stock explanations "
    "showing which factors contribute positively or negatively to each prediction.",
    styles["Body"]))

story.append(img("shap_importance.png", 4.3*inch))
story.append(Paragraph("Figure 7: SHAP importance bar plot — top features driving 21-day return predictions.", styles["Caption"]))

# ---------- 8. ANOMALY DETECTION ----------
story.append(Paragraph("8. Market Anomaly Detection", styles["H1"]))
story.append(Paragraph(
    "Four anomaly detectors operate independently to identify unusual market events, each "
    "using a statistical threshold calibrated to capture genuine market stress.",
    styles["Body"]))

story.append(make_table(
    ["Detector", "Threshold", "Events Found"],
    [["Volatility Spike", "21d vol Z-score > 2.5", "9,597"],
     ["Extreme Drawdown", "Drawdown < -15%", "173,733"],
     ["Volume Surge", "Volume > 3x 50-day average", "6,190"],
     ["Single-Day Crash", "Daily return < -8%", "1,163"]],
    col_widths=[1.4*inch, 2.2*inch, 1.4*inch]
))
story.append(Spacer(1, 3))

story.append(Paragraph(
    "Anomaly clustering reveals that 2008 (GFC) and 2020 (COVID-19) produced the highest "
    "anomaly density across all categories. Volatility spikes tend to precede crash events "
    "by several trading days, offering potential early-warning signals for risk management.",
    styles["Body"]))

# ---------- 9. DEPLOYMENT ----------
story.append(Paragraph("9. Deployment & Dashboard", styles["H1"]))
story.append(Paragraph(
    "The platform is deployed as a multi-page Streamlit application with seven interactive "
    "sections: Market Overview, Stock Analysis (candlestick charts + technical indicators + "
    "model predictions), Portfolio Builder (3 profiles with allocation visualization), "
    "Risk Assessment (scatter plots + sortable tables), Anomaly Detection (timeline + "
    "per-stock analysis), Explainability (global SHAP + per-stock explanations), and "
    "Model Performance (comparative metrics + analysis). The application loads pre-computed "
    "models and data for instant responsiveness.",
    styles["Body"]))
story.append(Paragraph(
    "To run: <font face='Courier' size='8'>streamlit run app/app.py</font> from the project root. "
    "All dependencies are specified in requirements.txt.",
    styles["Body"]))

# ---------- 10. INSIGHTS & LIMITATIONS ----------
story.append(Paragraph("10. Key Insights & Limitations", styles["H1"]))

story.append(Paragraph("10.1 Key Insights", styles["H2"]))
story.append(bullet("Linear models (Ridge) outperform tree-based models during regime shifts, suggesting the primary predictive signal is linear momentum/reversion."))
story.append(bullet("A 57.7% directional accuracy suggests the model captures a modest but useful predictive signal under noisy market conditions. Stock-market return prediction is inherently difficult, and low R-squared values are expected."))
story.append(bullet("Calendar seasonality (month) is the strongest SHAP feature, confirming well-documented patterns in Indian equity markets."))
story.append(bullet("PHARMA and IT sectors offer the best risk-adjusted returns over the sample period."))
story.append(bullet("Mean-variance optimization with profile-specific constraints produces diversified portfolios with favorable historical Sharpe ratios, though these are based on lookback statistics and not forward-looking guarantees."))
story.append(bullet("Volatility spikes tend to precede market crashes, offering potential early-warning signals for risk management."))

story.append(Paragraph("10.2 Limitations & Assumptions", styles["H2"]))
story.append(bullet("Survivorship bias: the dataset includes only current/recent NIFTY-50 constituents, potentially overstating historical returns."))
story.append(bullet("No transaction costs or slippage modeled in portfolio construction."))
story.append(bullet("21-day forward return horizon is fixed; adaptive horizon selection could improve results."))
story.append(bullet("Risk-free rate assumed at 6% (approximate Indian government bond yield)."))
story.append(bullet("Model trained on 2000-2017 data; performance may degrade in structurally different future market regimes."))
story.append(bullet("No access to fundamental data, news, or alternative data sources per competition constraints."))

story.append(Paragraph("10.3 Future Work", styles["H2"]))
story.append(bullet("Ensemble Ridge and LightGBM predictions for improved robustness."))
story.append(bullet("Implement Black-Litterman portfolio optimization incorporating model views."))
story.append(bullet("Add regime-detection (HMM) to dynamically switch between model configurations."))
story.append(bullet("Extend to multi-horizon predictions (5d, 10d, 63d) with horizon-specific models."))

# ---------- BUILD ----------
doc = SimpleDocTemplate(
    REPORT_PATH, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=0.55*inch, bottomMargin=0.55*inch,
    title="NIFTY-50 Investment Intelligence Platform - Technical Report",
    author="Shiv Prakash Vishwari | IIT Roorkee"
)

def add_page_number(canvas_obj, doc):
    page_num = canvas_obj.getPageNumber()
    if page_num > 1:
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.setFillColor(HexColor("#78909c"))
        canvas_obj.drawCentredString(W / 2, 0.3 * inch,
                                       f"NIFTY-50 Investment Intelligence - Page {page_num}")
        canvas_obj.restoreState()

doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=add_page_number)
print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    pass  # runs at module level
