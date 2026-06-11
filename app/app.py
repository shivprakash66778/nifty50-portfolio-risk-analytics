"""
app.py — NIFTY-50 Investment Intelligence Dashboard
════════════════════════════════════════════════════
Streamlit application with:
  1. Market Overview
  2. Stock Analysis & Prediction
  3. Portfolio Construction
  4. Risk Assessment
  5. Anomaly Detection
  6. Explainability (SHAP)
"""

import os, sys, json, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

warnings.filterwarnings("ignore")

# ── Resolve paths ────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.realpath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, PROJ_ROOT)

DATA_PATH = os.path.join(PROJ_ROOT, "data", "processed", "nifty50_features.parquet")
MODEL_DIR = os.path.join(PROJ_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Caching ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def load_risk_report():
    path = os.path.join(OUTPUT_DIR, "risk_report.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_portfolio(profile):
    path = os.path.join(OUTPUT_DIR, f"portfolio_{profile.lower()}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_anomalies():
    path = os.path.join(OUTPUT_DIR, "anomalies.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_model_results():
    path = os.path.join(OUTPUT_DIR, "model_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_feature_importance():
    path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_shap_importance():
    path = os.path.join(OUTPUT_DIR, "shap_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📈 NIFTY-50 Intelligence")
page = st.sidebar.radio("Navigate", [
    "🏠 Market Overview",
    "📊 Stock Analysis",
    "💼 Portfolio Builder",
    "⚠️ Risk Assessment",
    "🔍 Anomaly Detection",
    "🧠 Explainability",
    "📋 Model Performance",
])

df = load_data()
if df is None:
    st.error(
        "❌ Processed data not found at `data/processed/nifty50_features.parquet`.\n\n"
        "Run the full pipeline first:\n```\npython src/data_processing.py\n```"
    )
    st.stop()
symbols = sorted(df["Symbol"].unique())
industries = sorted(df["Industry"].dropna().unique())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Market Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Market Overview":
    st.title("NIFTY-50 Market Overview")
    st.markdown("Historical analysis of India's top 50 stocks (2000–2021)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks", df["Symbol"].nunique())
    col2.metric("Industries", df["Industry"].nunique())
    col3.metric("Date Range", f"{df['Date'].min().strftime('%Y')}–{df['Date'].max().strftime('%Y')}")
    col4.metric("Total Records", f"{len(df):,}")

    # Sector distribution
    st.subheader("Sector Distribution")
    sector_counts = df.drop_duplicates("Symbol")["Industry"].value_counts()
    fig = px.pie(values=sector_counts.values, names=sector_counts.index,
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Index proxy (equal-weighted)
    st.subheader("Market Trend (Equal-Weighted Index Proxy)")
    prices = df.pivot_table(index="Date", columns="Symbol", values="Close")
    normed = prices.divide(prices.iloc[0]).mean(axis=1)
    fig = px.line(x=normed.index, y=normed.values, labels={"x": "Date", "y": "Index (normalized)"})
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Sector performance heatmap
    st.subheader("Sector Annual Returns Heatmap")
    df_sub = df[df["Date"] >= "2005-01-01"].copy()
    df_sub["Year"] = df_sub["Date"].dt.year
    annual = df_sub.groupby(["Year", "Industry"]).apply(
        lambda g: g.sort_values("Date")["Close"].iloc[-1] / g.sort_values("Date")["Close"].iloc[0] - 1
        if len(g) > 10 else np.nan
    ).reset_index(name="Return")
    pivot = annual.pivot(index="Industry", columns="Year", values="Return")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                    labels=dict(color="Return"))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("Stock Return Correlation (Recent Period)")
    recent = df[df["Date"] >= "2018-01-01"]
    rets = recent.pivot_table(index="Date", columns="Symbol", values="ret_1d")
    top20 = rets.std().nsmallest(20).index  # top 20 least volatile for readability
    corr = rets[top20].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Stock Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Stock Analysis":
    st.title("Individual Stock Analysis")

    selected = st.sidebar.selectbox("Select Stock", symbols, index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0)
    stock_data = df[df["Symbol"] == selected].sort_values("Date")
    meta = stock_data[["Company_Name", "Industry"]].iloc[0]

    st.markdown(f"### {meta['Company_Name']} ({selected}) — {meta['Industry']}")

    col1, col2, col3, col4 = st.columns(4)
    latest = stock_data.iloc[-1]
    col1.metric("Last Close", f"₹{latest['Close']:.2f}")
    col2.metric("1M Return", f"{latest.get('ret_21d', 0):.2%}")
    col3.metric("RSI-14", f"{latest.get('RSI_14', 0):.1f}")
    col4.metric("21d Vol (ann.)", f"{latest.get('vol_21d', 0):.2%}")

    # Candlestick chart
    st.subheader("Price History")
    date_range = st.slider(
        "Date Range", min_value=stock_data["Date"].min().to_pydatetime(),
        max_value=stock_data["Date"].max().to_pydatetime(),
        value=(pd.Timestamp("2018-01-01").to_pydatetime(),
               stock_data["Date"].max().to_pydatetime()),
    )
    mask = (stock_data["Date"] >= date_range[0]) & (stock_data["Date"] <= date_range[1])
    view = stock_data[mask]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(
        x=view["Date"], open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"], name="OHLC",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["Date"], y=view["SMA_50"], name="SMA 50",
                             line=dict(color="orange", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["Date"], y=view["SMA_200"], name="SMA 200",
                             line=dict(color="blue", width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=view["Date"], y=view["Volume"], name="Volume",
                         marker_color="rgba(100,100,255,0.3)"), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Technical indicators
    st.subheader("Technical Indicators")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        fig = px.line(view, x="Date", y="RSI_14", title="RSI-14")
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with tcol2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["Date"], y=view["MACD_line"], name="MACD"))
        fig.add_trace(go.Scatter(x=view["Date"], y=view["MACD_signal"], name="Signal"))
        fig.add_trace(go.Bar(x=view["Date"], y=view["MACD_hist"], name="Histogram",
                             marker_color=np.where(view["MACD_hist"] >= 0, "green", "red")))
        fig.update_layout(title="MACD", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Prediction
    st.subheader("Model Prediction")
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "lgb_reg.pkl"))
        from src.prediction_engine import FEATURE_COLS
        latest_features = stock_data.dropna(subset=FEATURE_COLS).iloc[-1][FEATURE_COLS].values.reshape(1, -1)
        pred = model.predict(latest_features)[0]
        direction = "📈 Bullish" if pred > 0 else "📉 Bearish"
        st.info(f"**21-Day Return Forecast:** {pred:.2%} — {direction}")
    except Exception as e:
        st.warning(f"Could not load prediction model: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Portfolio Builder
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio Builder":
    st.title("Portfolio Construction")
    st.markdown("Mean-variance optimized portfolios for three investor profiles.")

    profile = st.selectbox("Select Investor Profile",
                           ["Conservative", "Balanced", "Aggressive"])

    port = load_portfolio(profile)
    if port is not None:
        # Portfolio metrics
        st.subheader(f"{profile} Portfolio")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("Stocks", len(port))
        pcol2.metric("Expected Return", f"{port['Ann_Return'].mean():.2%}")
        pcol3.metric("Avg Sharpe", f"{port['Sharpe'].mean():.2f}")
        pcol4.metric("Avg Max DD", f"{port['Max_DD'].mean():.2%}")

        # Allocation pie chart
        fig = px.pie(port, values="Weight_pct", names="Symbol",
                     title="Portfolio Allocation",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

        # Sector allocation
        sector_alloc = port.groupby("Industry")["Weight_pct"].sum().reset_index()
        fig = px.bar(sector_alloc, x="Industry", y="Weight_pct",
                     title="Sector Allocation", color="Industry")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Holdings table
        st.subheader("Holdings Detail")
        display_cols = ["Symbol", "Weight_pct", "Industry", "Ann_Return", "Ann_Vol", "Sharpe", "Max_DD"]
        st.dataframe(port[display_cols].style.format({
            "Weight_pct": "{:.2f}%", "Ann_Return": "{:.2%}",
            "Ann_Vol": "{:.2%}", "Sharpe": "{:.2f}", "Max_DD": "{:.2%}",
        }), use_container_width=True)

        # Compare all 3 profiles
        st.subheader("Profile Comparison")
        comp_data = []
        for p in ["Conservative", "Balanced", "Aggressive"]:
            pf = load_portfolio(p)
            if pf is not None:
                comp_data.append({
                    "Profile": p,
                    "Stocks": len(pf),
                    "Avg Return": pf["Ann_Return"].mean(),
                    "Avg Vol": pf["Ann_Vol"].mean(),
                    "Avg Sharpe": pf["Sharpe"].mean(),
                })
        if comp_data:
            st.dataframe(pd.DataFrame(comp_data).style.format({
                "Avg Return": "{:.2%}", "Avg Vol": "{:.2%}", "Avg Sharpe": "{:.2f}",
            }), use_container_width=True)
    else:
        st.warning("Portfolio data not found. Run src/portfolio.py first.")
      
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Assessment":
    st.title("Risk Assessment")

    risk_df = load_risk_report()

    if risk_df is not None and len(risk_df) > 0:
        st.subheader("Risk-Return Scatter")

        risk_plot_df = risk_df.copy()

        # Convert important columns to numeric safely
        numeric_cols = [
            "Ann_Volatility",
            "Ann_Return",
            "Sharpe_Ratio",
            "Max_Drawdown",
            "Beta",
            "VaR_95"
        ]

        for col in numeric_cols:
            if col in risk_plot_df.columns:
                risk_plot_df[col] = pd.to_numeric(risk_plot_df[col], errors="coerce")

        # Drop rows where essential plotting values are missing
        risk_plot_df = risk_plot_df.dropna(
            subset=["Ann_Volatility", "Ann_Return", "Sharpe_Ratio"]
        )

        # Plotly marker size cannot be negative.
        # Sharpe_Ratio can be negative, so create safe positive size.
        risk_plot_df["Sharpe_Size"] = risk_plot_df["Sharpe_Ratio"].clip(lower=0)

        # If all Sharpe ratios are <= 0, use constant marker size
        if risk_plot_df["Sharpe_Size"].max() == 0:
            risk_plot_df["Sharpe_Size"] = 1

        # Make sure marker size is not too tiny
        risk_plot_df["Sharpe_Size"] = risk_plot_df["Sharpe_Size"] + 0.1

        hover_cols = [c for c in ["Max_Drawdown", "Beta"] if c in risk_plot_df.columns]

        fig = px.scatter(
            risk_plot_df,
            x="Ann_Volatility",
            y="Ann_Return",
            text="Name" if "Name" in risk_plot_df.columns else None,
            size="Sharpe_Size",
            color="Sharpe_Ratio",
            color_continuous_scale="RdYlGn",
            hover_data=hover_cols,
            title="Risk-Return Profile of NIFTY-50 Stocks"
        )

        fig.update_traces(textposition="top center", textfont_size=8)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top/bottom tables
        st.subheader("Top 10 by Sharpe Ratio")

        cols = [
            "Name",
            "Ann_Return",
            "Ann_Volatility",
            "Sharpe_Ratio",
            "Sortino_Ratio",
            "Max_Drawdown",
            "VaR_95",
            "Beta"
        ]

        valid_cols = [c for c in cols if c in risk_df.columns]

        if "Sharpe_Ratio" in risk_df.columns:
            st.dataframe(
                risk_df.sort_values("Sharpe_Ratio", ascending=False)
                .head(10)[valid_cols],
                use_container_width=True
            )
        else:
            st.warning("Sharpe_Ratio column not found in risk report.")

        st.subheader("Riskiest 10 Stocks (by Max Drawdown)")

        if "Max_Drawdown" in risk_df.columns:
            st.dataframe(
                risk_df.sort_values("Max_Drawdown")
                .head(10)[valid_cols],
                use_container_width=True
            )
        else:
            st.warning("Max_Drawdown column not found in risk report.")

        # VaR distribution
        st.subheader("Value at Risk (95%) Distribution")

        if "VaR_95" in risk_df.columns:
            fig = px.histogram(
                risk_df,
                x="VaR_95",
                nbins=30,
                title="Daily VaR(95%) Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("VaR_95 column not found in risk report.")

    else:
        st.warning("Risk data not found. Run src/risk_assessment.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    st.title("Market Anomaly Detection")

    anomalies = load_anomalies()
    if anomalies is not None and len(anomalies) > 0:
        anomalies["Date"] = pd.to_datetime(anomalies["Date"])

        st.subheader("Anomaly Summary")
        summary = anomalies["anomaly_type"].value_counts()
        fig = px.bar(x=summary.index, y=summary.values,
                     labels={"x": "Type", "y": "Count"},
                     color=summary.index)
        st.plotly_chart(fig, use_container_width=True)

        # Timeline
        st.subheader("Anomaly Timeline")
        anom_type = st.selectbox("Anomaly Type", anomalies["anomaly_type"].unique())
        sub = anomalies[anomalies["anomaly_type"] == anom_type]
        yearly = sub.groupby(sub["Date"].dt.year).size().reset_index(name="Count")
        yearly.columns = ["Year", "Count"]
        fig = px.bar(yearly, x="Year", y="Count", title=f"{anom_type} by Year")
        st.plotly_chart(fig, use_container_width=True)

        # Per stock
        st.subheader("Most Anomalous Stocks")
        top_stocks = sub["Symbol"].value_counts().head(15)
        fig = px.bar(x=top_stocks.index, y=top_stocks.values,
                     labels={"x": "Stock", "y": "Events"})
        st.plotly_chart(fig, use_container_width=True)

        # Recent anomalies table
        st.subheader("Recent Anomaly Events")
        st.dataframe(sub.sort_values("Date", ascending=False).head(50),
                     use_container_width=True)
    else:
        st.warning("No anomaly data found. Run src/anomaly_detection.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Explainability
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.title("Explainable AI — Why the Model Recommends")

    shap_imp = load_shap_importance()
    fi = load_feature_importance()

    if shap_imp is not None:
        st.subheader("SHAP Feature Importance")
        fig = px.bar(shap_imp.head(15), x="mean_abs_shap", y="feature",
                     orientation="h", title="Mean |SHAP| Values")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

    # SHAP plots from saved images
    for img_name, title in [("shap_importance.png", "SHAP Bar Plot"),
                            ("shap_summary.png", "SHAP Summary Plot")]:
        path = os.path.join(CHART_DIR, img_name)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)

    if fi is not None:
        st.subheader("LightGBM Feature Importance (Gain)")
        fig = px.bar(fi.head(15), x="importance", y="feature",
                     orientation="h", title="Feature Importance (Split Count)")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Single stock explanation
    st.subheader("Explain Single Stock Prediction")
    sel = st.selectbox("Stock", symbols, key="explain_stock")
    
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "lgb_reg.pkl"))
    
        import shap
        from src.prediction_engine import FEATURE_COLS
    
        stock_df = df[df["Symbol"] == sel].copy()
    
        # Keep only required feature columns that exist
        available_cols = [c for c in FEATURE_COLS if c in stock_df.columns]
    
        if len(available_cols) == 0:
            st.warning("No model feature columns found in dataset.")
        else:
            # Convert feature columns to numeric before dropna/model prediction
            for col in available_cols:
                stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")
    
            stock_df = stock_df.dropna(subset=available_cols).sort_values("Date")
    
            if len(stock_df) > 0:
                # Create one-row feature dataframe with correct numeric dtype
                X_df = stock_df.iloc[[-1]][available_cols].copy()
                X_df = X_df.apply(pd.to_numeric, errors="coerce")
                X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
                X_df = X_df.astype("float64")
    
                pred = model.predict(X_df)[0]
                st.info(f"Predicted 21-day return: **{pred:.2%}**")
    
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_df)
    
                # Handle SHAP output safely
                if isinstance(sv, list):
                    shap_values = sv[0][0]
                else:
                    shap_values = sv[0]
    
                exp_df = pd.DataFrame({
                    "Feature": available_cols,
                    "Value": X_df.iloc[0].values,
                    "SHAP Contribution": shap_values
                })
    
                exp_df["Abs Contribution"] = exp_df["SHAP Contribution"].abs()
                exp_df = exp_df.sort_values("Abs Contribution", ascending=False)
    
                fig = px.bar(
                    exp_df.head(10),
                    x="SHAP Contribution",
                    y="Feature",
                    orientation="h",
                    color="SHAP Contribution",
                    color_continuous_scale="RdYlGn",
                    title=f"Top Factors for {sel} Prediction"
                )
    
                fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
                st.plotly_chart(fig, use_container_width=True)
    
                st.dataframe(
                    exp_df.head(10)[["Feature", "Value", "SHAP Contribution"]].style.format({
                        "Value": "{:.4f}",
                        "SHAP Contribution": "{:.6f}"
                    }),
                    use_container_width=True
                )
            else:
                st.warning("Not enough valid numeric feature data found for this stock.")
    
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Model Performance":
    st.title("Model Performance Summary")

    results = load_model_results()
    if results:
        st.subheader("Regression Models (21-Day Return Prediction)")
        reg = pd.DataFrame(results["regression"]).T
        st.dataframe(reg.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Classification Models (Direction Prediction)")
        clf = pd.DataFrame(results["classification"]).T
        st.dataframe(clf.style.format("{:.4f}"), use_container_width=True)

        # Visual comparison
        st.subheader("Model Comparison")
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Test RMSE (lower=better)",
                                                             "Test Dir. Accuracy (higher=better)"])
        models = list(results["regression"].keys())
        rmses = [results["regression"][m]["test_RMSE"] for m in models]
        daccs = [results["regression"][m]["test_DirAcc"] for m in models]

        fig.add_trace(go.Bar(x=models, y=rmses, name="RMSE",
                             marker_color=["#636EFA", "#EF553B", "#00CC96"]), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=daccs, name="Dir. Acc",
                             marker_color=["#636EFA", "#EF553B", "#00CC96"]), row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key Observations:**
        - Ridge achieves competitive performance despite simplicity, suggesting returns have a significant linear component
        - Tree-based models (XGBoost, LightGBM) capture non-linear patterns but risk overfitting in volatile markets
        - ~57% directional accuracy suggests a modest but useful predictive signal under noisy market conditions
        - Low R² is expected in financial return forecasting; directional accuracy is more meaningful for decision support
        - Test period (2019–2020) includes COVID-19 regime shift, making it a challenging evaluation
        """)
    else:
        st.warning("No model results found. Run src/prediction_engine.py first.")


# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Built for Cult Open Projects 2026**")
st.sidebar.markdown("NIFTY-50 Investment Intelligence Platform")
