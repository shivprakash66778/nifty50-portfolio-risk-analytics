"""
prediction_engine.py
────────────────────
Train and evaluate stock prediction models using time-based splits.
Models: LightGBM (regression + classification), XGBoost, Ridge baseline.
Outputs: trained models, evaluation metrics, feature importances.
"""

import os, warnings, json, joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, classification_report
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(PROJ_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature columns ──────────────────────────────────────────────────────────
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

REG_TARGET = "fwd_ret_21d"
CLF_TARGET = "fwd_direction"


def prepare_data(df: pd.DataFrame):
    """Time-based train/val/test split.
    Train:  < 2018-01-01
    Val:    2018-01-01 – 2019-06-30
    Test:   2019-07-01 – 2021-01-01 (leaving last ~4 months for forward labels)
    """
    df = df.dropna(subset=FEATURE_COLS + [REG_TARGET, CLF_TARGET]).copy()

    train = df[df["Date"] < "2018-01-01"]
    val   = df[(df["Date"] >= "2018-01-01") & (df["Date"] < "2019-07-01")]
    test  = df[(df["Date"] >= "2019-07-01") & (df["Date"] < "2021-01-01")]

    return train, val, test


def train_regression_models(train, val, test):
    """Train Ridge, XGBoost, LightGBM regressors."""
    X_tr, y_tr = train[FEATURE_COLS], train[REG_TARGET]
    X_val, y_val = val[FEATURE_COLS], val[REG_TARGET]
    X_te, y_te = test[FEATURE_COLS], test[REG_TARGET]

    results = {}

    # ── Ridge Baseline ──
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    pred_val = ridge.predict(X_val)
    pred_te = ridge.predict(X_te)
    results["Ridge"] = _eval_reg(y_val, pred_val, y_te, pred_te)
    joblib.dump(ridge, os.path.join(MODEL_DIR, "ridge_reg.pkl"))

    # ── XGBoost ──
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        early_stopping_rounds=30, random_state=42, n_jobs=-1,
        tree_method="hist"
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred_val = xgb_model.predict(X_val)
    pred_te = xgb_model.predict(X_te)
    results["XGBoost"] = _eval_reg(y_val, pred_val, y_te, pred_te)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_reg.pkl"))

    # ── LightGBM ──
    lgb_model = lgb.LGBMRegressor(
        n_estimators=800, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05,
        num_leaves=63, min_child_samples=50,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )
    pred_val = lgb_model.predict(X_val)
    pred_te = lgb_model.predict(X_te)
    results["LightGBM"] = _eval_reg(y_val, pred_val, y_te, pred_te)
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgb_reg.pkl"))

    # Save feature importances
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": lgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    return results, lgb_model


def train_classification_models(train, val, test):
    """Train direction classifiers."""
    X_tr, y_tr = train[FEATURE_COLS], train[CLF_TARGET]
    X_val, y_val = val[FEATURE_COLS], val[CLF_TARGET]
    X_te, y_te = test[FEATURE_COLS], test[CLF_TARGET]

    results = {}

    # ── Logistic Baseline ──
    lr = LogisticRegression(max_iter=500, C=0.1)
    lr.fit(X_tr, y_tr)
    pred_val = lr.predict_proba(X_val)[:, 1]
    pred_te = lr.predict_proba(X_te)[:, 1]
    results["LogReg"] = _eval_clf(y_val, pred_val, y_te, pred_te)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logreg_clf.pkl"))

    # ── LightGBM Classifier ──
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        num_leaves=50, min_child_samples=50,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )
    pred_val = lgb_clf.predict_proba(X_val)[:, 1]
    pred_te = lgb_clf.predict_proba(X_te)[:, 1]
    results["LightGBM_clf"] = _eval_clf(y_val, pred_val, y_te, pred_te)
    joblib.dump(lgb_clf, os.path.join(MODEL_DIR, "lgb_clf.pkl"))

    return results, lgb_clf


def _eval_reg(y_val, p_val, y_te, p_te):
    return {
        "val_MAE": round(mean_absolute_error(y_val, p_val), 6),
        "val_RMSE": round(np.sqrt(mean_squared_error(y_val, p_val)), 6),
        "val_R2": round(r2_score(y_val, p_val), 4),
        "test_MAE": round(mean_absolute_error(y_te, p_te), 6),
        "test_RMSE": round(np.sqrt(mean_squared_error(y_te, p_te)), 6),
        "test_R2": round(r2_score(y_te, p_te), 4),
        "val_DirAcc": round((np.sign(p_val) == np.sign(y_val.values)).mean(), 4),
        "test_DirAcc": round((np.sign(p_te) == np.sign(y_te.values)).mean(), 4),
    }


def _eval_clf(y_val, p_val, y_te, p_te, threshold=0.5):
    return {
        "val_AUC": round(roc_auc_score(y_val, p_val), 4),
        "val_Acc": round(accuracy_score(y_val, (p_val > threshold).astype(int)), 4),
        "test_AUC": round(roc_auc_score(y_te, p_te), 4),
        "test_Acc": round(accuracy_score(y_te, (p_te > threshold).astype(int)), 4),
    }


def run_prediction_pipeline(df: pd.DataFrame):
    """Full prediction pipeline."""
    print("Preparing time-based splits...")
    train, val, test = prepare_data(df)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"  Train period: {train['Date'].min().date()} – {train['Date'].max().date()}")
    print(f"  Val period:   {val['Date'].min().date()} – {val['Date'].max().date()}")
    print(f"  Test period:  {test['Date'].min().date()} – {test['Date'].max().date()}")

    print("\n── Regression Models (predict 21-day return) ──")
    reg_results, best_reg = train_regression_models(train, val, test)
    for name, metrics in reg_results.items():
        print(f"  {name}: test RMSE={metrics['test_RMSE']:.4f}, "
              f"R²={metrics['test_R2']:.4f}, DirAcc={metrics['test_DirAcc']:.4f}")

    print("\n── Classification Models (predict direction) ──")
    clf_results, best_clf = train_classification_models(train, val, test)
    for name, metrics in clf_results.items():
        print(f"  {name}: test AUC={metrics['test_AUC']:.4f}, Acc={metrics['test_Acc']:.4f}")

    # Save results
    all_results = {"regression": reg_results, "classification": clf_results}
    with open(os.path.join(OUTPUT_DIR, "model_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nModels saved. Results saved to outputs/model_results.json")
    return all_results, best_reg, best_clf, test


if __name__ == "__main__":
    data_path = os.path.join(PROJ_ROOT, "data", "processed", "nifty50_features.parquet")
    df = pd.read_parquet(data_path)
    run_prediction_pipeline(df)
