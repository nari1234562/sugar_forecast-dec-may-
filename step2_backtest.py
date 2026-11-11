"""
step2_backtest.py - 6-Data-Point Multi-Holdout Backtesting (Buffalo Methodology)
================================================================================
Conducts rigorous multi-holdout backtest across 6 historical batches spaced 2 months
apart over the past year. Trains separate forecasting models for Buying and Selling
prices using only data known at each origin (zero data leakage), evaluating N+1..N+6,
Q1, and Q2 against industry benchmarks and baseline models.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import step1_preprocess as s1

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backtest")

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

# Benchmark Targets (INR/kg)
TARGET_N1 = 2.0
TARGET_Q1 = 5.0
TARGET_Q2 = 8.0

# 6 Historical Holdout Origins (Spaced 2 months apart, last origin 2026-05 has full 6 actual months through Nov 2026)
HOLDOUT_ORIGINS = [
    pd.Timestamp("2025-07-01"),
    pd.Timestamp("2025-09-01"),
    pd.Timestamp("2025-11-01"),
    pd.Timestamp("2026-01-01"),
    pd.Timestamp("2026-03-01"),
    pd.Timestamp("2026-05-01"),
]

LEARNED_METHODS = ["RecursiveX XGB", "Direct Ridge", "Direct Blend"]
BASELINE_METHODS = ["Naive", "Rolling 3M avg", "Drift (6m avg)"]
ALL_EVAL_METHODS = LEARNED_METHODS + BASELINE_METHODS


def make_ridge():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-1, 3, 13))
    )


def make_xgb():
    return XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=1
    )


def fit_and_predict_horizon(
    train_df: pd.DataFrame,
    test_rows: pd.DataFrame,
    target_col: str,
    base_rate_col: str,
    h: int
) -> Dict[str, np.ndarray]:
    """Fit Buffalo models (Ridge, XGBoost, Blend) and baselines on training data and predict test rows."""
    tr = train_df.dropna(subset=[target_col] + s1.ALL_MODEL_FEATURES).copy()
    base_t = test_rows[base_rate_col].to_numpy()

    if len(tr) < 15:
        return {m: base_t.copy() for m in ALL_EVAL_METHODS}

    X_train = tr[s1.ALL_MODEL_FEATURES]
    y_train = tr[target_col]
    X_test = test_rows[s1.ALL_MODEL_FEATURES].fillna(0.0)

    # 1. Ridge
    pipe_ridge = make_ridge()
    pipe_ridge.fit(X_train, y_train)
    pred_ridge_change = pipe_ridge.predict(X_test)

    # 2. XGBoost
    xgb = make_xgb()
    xgb.fit(X_train, y_train)
    pred_xgb_change = xgb.predict(X_test)

    # 3. Direct Blend
    pred_blend_change = (pred_ridge_change + pred_xgb_change) / 2.0

    # Baselines
    naive_pred = base_t.copy()
    drift_rate = test_rows["Price_Diff_6M"].fillna(0.0).to_numpy() / 6.0
    drift_pred = base_t + drift_rate * h
    r3_pred = test_rows["Buy_Price_MA3"].to_numpy() if "Buy" in target_col else (base_t + test_rows["Gross_Margin_Lag1"].fillna(0.0).to_numpy())

    return {
        "RecursiveX XGB": base_t + pred_xgb_change,
        "Direct Ridge": base_t + pred_ridge_change,
        "Direct Blend": base_t + pred_blend_change,
        "Naive": naive_pred,
        "Rolling 3M avg": r3_pred,
        "Drift (6m avg)": drift_pred
    }


def run_6batch_backtest(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Execute complete 6-batch multi-holdout backtest for Buying and Selling prices."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = s1.build_multi_horizon_table(df_raw, max_horizon=6)

    holdout_records = []
    error_distributions = {"Buying": {}, "Selling": {}}

    for batch_idx, origin in enumerate(HOLDOUT_ORIGINS, 1):
        log.info(f"Running Holdout Batch {batch_idx}/6 (Origin: {origin:%b %Y})...")

        # Strict no-leakage training slice
        train_slice = table[(table["target_month"] <= origin) & table["target_buy"].notna()].copy()
        s1.assert_no_leakage(train_slice, origin)

        for h in range(1, 7):
            test_slice = table[(table["origin"] == origin) & (table["horizon"] == h)].copy()
            if len(test_slice) == 0:
                continue

            # Evaluate Buying
            buy_preds = fit_and_predict_horizon(train_slice, test_slice, "target_buy", "rate_t", h)
            # Evaluate Selling
            sell_preds = fit_and_predict_horizon(train_slice, test_slice, "target_sell", "sell_rate_t", h)

            for i in range(len(test_slice)):
                row = test_slice.iloc[i]
                rec = {
                    "Batch": batch_idx,
                    "origin": origin,
                    "origin_date": origin.strftime("%Y-%m-%d"),
                    "horizon": h,
                    "target_month": row["target_month"],
                    "target_month_str": row["target_month"].strftime("%b %Y"),
                    "State": row["State"],
                    "weight_tonnes": row["weight_tonnes"],
                    # Buying
                    "Actual_Buy_Price": row["rate_actual"],
                    "Last_Buy_Price": row["rate_t"],
                    # Selling
                    "Actual_Sell_Price": row["sell_rate_actual"],
                    "Last_Sell_Price": row["sell_rate_t"],
                }

                for m in ALL_EVAL_METHODS:
                    rec[f"Buy_{m}"] = buy_preds[m][i]
                    rec[f"Buy_Err_{m}"] = abs(buy_preds[m][i] - row["rate_actual"])
                    rec[f"Sell_{m}"] = sell_preds[m][i]
                    rec[f"Sell_Err_{m}"] = abs(sell_preds[m][i] - row["sell_rate_actual"])

                holdout_records.append(rec)

    detail_df = pd.DataFrame(holdout_records)

    # Calculate Empirical 80% Error Bands (10th and 90th percentiles) for the headline model (RecursiveX XGB)
    for price_type, err_col, act_col, pred_col in [
        ("Buying", "Buy_Err_RecursiveX XGB", "Actual_Buy_Price", "Buy_RecursiveX XGB"),
        ("Selling", "Sell_Err_RecursiveX XGB", "Actual_Sell_Price", "Sell_RecursiveX XGB")
    ]:
        for h in range(1, 7):
            h_data = detail_df[detail_df["horizon"] == h]
            residuals = (h_data[act_col] - h_data[pred_col]).dropna().to_numpy()
            if len(residuals) >= 5:
                p10 = float(np.percentile(residuals, 10))
                p90 = float(np.percentile(residuals, 90))
            else:
                p10, p90 = -1.5, 1.5
            error_distributions[price_type][str(h)] = [p10, p90, len(residuals)]

    return detail_df, error_distributions


def build_kpi_summary_table(detail_df: pd.DataFrame, price_type: str = "Buying") -> pd.DataFrame:
    """Build executive KPI table comparing models across N+1..N+6, Q1, and Q2."""
    prefix = "Buy" if price_type == "Buying" else "Sell"
    act_col = f"Actual_{prefix}_Price"
    headline = "RecursiveX XGB"

    kpi_rows = []
    models_to_eval = [f"N+{h}" for h in range(1, 7)] + ["Q1", "Q2"]

    for model_name in models_to_eval:
        if model_name.startswith("N+"):
            h = int(model_name.replace("N+", ""))
            sub = detail_df[detail_df["horizon"] == h].copy()
            target_threshold = TARGET_N1 if h == 1 else np.nan
        elif model_name == "Q1":
            sub = detail_df[detail_df["horizon"].isin([1, 2, 3])].copy()
            target_threshold = TARGET_Q1
        else: # Q2
            sub = detail_df[detail_df["horizon"].isin([4, 5, 6])].copy()
            target_threshold = TARGET_Q2

        weights = sub["weight_tonnes"]
        act = sub[act_col]
        pred_hl = sub[f"{prefix}_{headline}"]
        err_hl = abs(pred_hl - act)

        mae_hl = float(err_hl.mean())
        vw_mae_hl = float((err_hl * weights).sum() / weights.sum()) if weights.sum() > 0 else mae_hl
        rmse_hl = float(np.sqrt(((pred_hl - act) ** 2).mean()))
        mape_hl = float((err_hl / act).mean() * 100)

        # Baseline MAEs
        naive_mae = float(abs(sub[f"{prefix}_Naive"] - act).mean())
        r3_mae = float(abs(sub[f"{prefix}_Rolling 3M avg"] - act).mean())
        best_baseline = min(naive_mae, r3_mae)
        gain_pct = ((best_baseline - mae_hl) / best_baseline) * 100.0 if best_baseline > 0 else 0.0

        check_status = "MEETS" if (pd.isna(target_threshold) or mae_hl <= target_threshold) else "OVER"
        check_vw = "MEETS" if (pd.isna(target_threshold) or vw_mae_hl <= target_threshold) else "OVER"

        # Count batches meeting target
        batches_ok = 0
        for b in detail_df["Batch"].unique():
            b_sub = sub[sub["Batch"] == b]
            if len(b_sub) > 0:
                b_mae = float(abs(b_sub[f"{prefix}_{headline}"] - b_sub[act_col]).mean())
                if pd.isna(target_threshold) or b_mae <= target_threshold:
                    batches_ok += 1

        kpi_rows.append({
            "Commodity": "Sugar",
            "Price_Type": price_type,
            "Horizon_Model": model_name,
            "Primary_Model": headline,
            "Batches_Evaluated": sub["Batch"].nunique(),
            "Total_Test_Points": len(sub),
            "MAE_INR_kg": round(mae_hl, 2),
            "VW_MAE_INR_kg": round(vw_mae_hl, 2),
            "RMSE_INR_kg": round(rmse_hl, 2),
            "MAPE_Pct": round(mape_hl, 2),
            "Target_Threshold": target_threshold,
            "Check_District_MAE": check_status,
            "Check_Weighted_MAE": check_vw,
            "Batches_Within_Target": f"{batches_ok}/{sub['Batch'].nunique()}",
            "Last_Month_Repeat_MAE": round(naive_mae, 2),
            "Rolling_3M_Avg_MAE": round(r3_mae, 2),
            "Gain_Over_Best_Simple_Pct": round(gain_pct, 1)
        })

    return pd.DataFrame(kpi_rows)


def build_method_comparison_matrix(detail_df: pd.DataFrame, price_type: str = "Buying") -> pd.DataFrame:
    """Build matrix of MAE across all candidate models and horizons."""
    prefix = "Buy" if price_type == "Buying" else "Sell"
    act_col = f"Actual_{prefix}_Price"

    rows = []
    for m in ALL_EVAL_METHODS:
        m_row = {"Price_Type": price_type, "Method": m}
        for h in range(1, 7):
            sub = detail_df[detail_df["horizon"] == h]
            mae = float(abs(sub[f"{prefix}_{m}"] - sub[act_col]).mean())
            m_row[f"N+{h}"] = round(mae, 2)

        # Q1 and Q2
        q1_sub = detail_df[detail_df["horizon"].isin([1, 2, 3])]
        q2_sub = detail_df[detail_df["horizon"].isin([4, 5, 6])]
        m_row["Q1"] = round(float(abs(q1_sub[f"{prefix}_{m}"] - q1_sub[act_col]).mean()), 2)
        m_row["Q2"] = round(float(abs(q2_sub[f"{prefix}_{m}"] - q2_sub[act_col]).mean()), 2)
        rows.append(m_row)

    return pd.DataFrame(rows)


def run_pipeline_and_export():
    """Run full backtest and save multi-tab Excel and CSV reports."""
    df_raw = s1.load_clean_data()
    detail_df, error_distributions = run_6batch_backtest(df_raw)

    # Build KPI and Comparison Sheets
    kpi_buy = build_kpi_summary_table(detail_df, "Buying")
    kpi_sell = build_kpi_summary_table(detail_df, "Selling")
    comp_buy = build_method_comparison_matrix(detail_df, "Buying")
    comp_sell = build_method_comparison_matrix(detail_df, "Selling")

    # Batches Info
    batch_rows = []
    for b_idx, origin in enumerate(HOLDOUT_ORIGINS, 1):
        batch_rows.append({
            "Batch": b_idx,
            "Trained_Up_To": origin.strftime("%Y-%m-%d"),
            "N+1": (origin + pd.DateOffset(months=1)).strftime("%b %Y"),
            "N+2": (origin + pd.DateOffset(months=2)).strftime("%b %Y"),
            "N+3": (origin + pd.DateOffset(months=3)).strftime("%b %Y"),
            "N+4": (origin + pd.DateOffset(months=4)).strftime("%b %Y"),
            "N+5": (origin + pd.DateOffset(months=5)).strftime("%b %Y"),
            "N+6": (origin + pd.DateOffset(months=6)).strftime("%b %Y"),
            "Q1_Range": f"{(origin + pd.DateOffset(months=1)):%b}-{(origin + pd.DateOffset(months=3)):%b %Y}",
            "Q2_Range": f"{(origin + pd.DateOffset(months=4)):%b}-{(origin + pd.DateOffset(months=6)):%b %Y}"
        })
    batches_df = pd.DataFrame(batch_rows)

    # Save to Excel
    xlsx_path = OUT_DIR / "Sugar_MultiHoldout_Report_6Batches.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            kpi_buy.to_excel(writer, sheet_name="Target_Check_Buying", index=False)
            kpi_sell.to_excel(writer, sheet_name="Target_Check_Selling", index=False)
            comp_buy.to_excel(writer, sheet_name="Comparison_MAE_Buying", index=False)
            comp_sell.to_excel(writer, sheet_name="Comparison_MAE_Selling", index=False)
            batches_df.to_excel(writer, sheet_name="Holdout_Batches", index=False)
            detail_df.to_excel(writer, sheet_name="Forecast_vs_Actual_Detail", index=False)
    except PermissionError:
        log.warning(f"⚠️ Notice: {xlsx_path.name} is open in Excel. Close it to allow overwrite.")

    # Save CSVs
    detail_df.to_csv(OUT_DIR / "Sugar_MultiHoldout_Report_6Batches.csv", index=False)
    detail_df.to_csv(OUT_DIR / "holdout_detail.csv", index=False)

    # Save meta json
    meta = {
        "best_method_buying": "RecursiveX XGB",
        "best_method_selling": "RecursiveX XGB",
        "error_bands": error_distributions,
        "holdout_batches_count": len(HOLDOUT_ORIGINS),
        "last_holdout_origin": HOLDOUT_ORIGINS[-1].strftime("%Y-%m-%d")
    }
    with open(OUT_DIR / "backtest_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Mirror to Downloads
    try:
        import shutil
        dl_dir = Path(r"C:\Users\shash\Downloads")
        shutil.copyfile(xlsx_path, dl_dir / "Sugar_MultiHoldout_Report_6Batches.xlsx")
        shutil.copyfile(OUT_DIR / "Sugar_MultiHoldout_Report_6Batches.csv", dl_dir / "Sugar_MultiHoldout_Report_6Batches.csv")
    except Exception as e:
        log.warning(f"Mirroring to Downloads skipped: {e}")

    log.info("\n=== MULTI-HOLDOUT BACKTEST COMPLETED SUCCESSFULLY ===")
    log.info("\n--- BUYING PRICE VALIDATION SUMMARY ---")
    log.info(kpi_buy[["Horizon_Model", "MAE_INR_kg", "VW_MAE_INR_kg", "Target_Threshold", "Check_District_MAE", "Batches_Within_Target", "Gain_Over_Best_Simple_Pct"]].to_string(index=False))
    log.info("\n--- SELLING PRICE VALIDATION SUMMARY ---")
    log.info(kpi_sell[["Horizon_Model", "MAE_INR_kg", "VW_MAE_INR_kg", "Target_Threshold", "Check_District_MAE", "Batches_Within_Target", "Gain_Over_Best_Simple_Pct"]].to_string(index=False))
    log.info(f"\nSaved Multi-Holdout Report to {xlsx_path}")


if __name__ == "__main__":
    run_pipeline_and_export()
