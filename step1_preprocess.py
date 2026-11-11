"""
step1_preprocess.py - Data Preprocessing & Feature Pipeline for Staples Commodity Forecasting
=============================================================================================
Loads the verified clean Indian Sugar dataset across 5 states (Karnataka, Uttar Pradesh,
Gujarat, Maharashtra, West Bengal), performs zero-leakage multi-horizon table construction,
and provides standardized feature sets for both Buying Price and Selling Price models.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Default dataset paths
DEFAULT_DATA_XLSX = HERE / "data" / "2Sugar_Top5_States_Clean_Input_Dataset.xlsx"
DEFAULT_DATA_CSV = HERE / "data" / "2Sugar_Top5_States_Clean_Input_Dataset.csv"

STATES = ["Karnataka", "Uttar Pradesh", "Gujarat", "Maharashtra", "West Bengal"]

# Official state production weights for volume-weighted aggregation (Lakh Metric Tonnes)
STATE_WEIGHTS = {
    "Uttar Pradesh": 110.0,
    "Maharashtra": 105.0,
    "Karnataka": 45.0,
    "Gujarat": 12.0,
    "West Bengal": 10.0,
}

# Core feature definitions
PRICE_FEATURES = [
    "Buy_Price_Lag1", "Buy_Price_Lag2", "Buy_Price_Lag3",
    "Sell_Price_Lag1", "Gross_Margin_Lag1",
    "Price_Diff_1M", "Price_Diff_3M", "Price_Diff_6M",
    "Price_Acceleration",
    "Buy_Price_MA3", "Buy_Price_MA6", "Price_Dev_MA3",
    "Price_Volatility_3M"
]

WEATHER_FEATURES = [
    "Avg_Temperature_C", "Max_Temperature_C", "Min_Temperature_C",
    "Avg_Relative_Humidity_Pct", "Total_Precipitation_mm",
    "Heat_Stress_Flag", "Dry_Spell_Flag",
    "Temp_Range_C", "Rainfall_3M_Cum"
]

APY_FEATURES = [
    "Sugarcane_Area_Lakh_Ha", "Sugarcane_Production_Lakh_Tonnes",
    "Sugarcane_Yield_Kg_Per_Ha", "Sugar_Recovery_Rate_Pct",
    "Monthly_Sugar_Produced_Lakh_Tonnes"
]

SEASONALITY_FEATURES = [
    "Crushing_Season_Flag", "Month_sin", "Month_cos"
]

ALL_MODEL_FEATURES = PRICE_FEATURES + WEATHER_FEATURES + APY_FEATURES + SEASONALITY_FEATURES


def find_dataset_file(custom_path: str | Path | None = None) -> Path:
    """Locate the dataset file supporting both 2Sugar_* and Sugar_* naming and XLSX/CSV formats."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    data_dir = HERE / "data"
    candidates = [
        data_dir / "2Sugar_Top5_States_Clean_Input_Dataset.xlsx",
        data_dir / "2Sugar_Top5_States_Clean_Input_Dataset.csv",
        data_dir / "Sugar_Top5_States_Clean_Input_Dataset.xlsx",
        data_dir / "Sugar_Top5_States_Clean_Input_Dataset.csv",
        Path(r"C:\Users\shash\Downloads\2Sugar_Top5_States_Clean_Input_Dataset.xlsx"),
        Path(r"C:\Users\shash\Downloads\Sugar_Top5_States_Clean_Input_Dataset.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Clean input dataset not found in {data_dir}. Expected 2Sugar_Top5_States_Clean_Input_Dataset.xlsx or .csv.")


def load_clean_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the clean 5-state sugar input dataset from XLSX or CSV."""
    p = find_dataset_file(csv_path)

    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    df["Month_Start_Date"] = pd.to_datetime(df["Month_Start_Date"])
    df.sort_values(by=["State", "Month_Start_Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Validate state coverage
    present_states = set(df["State"].unique())
    missing_states = set(STATES) - present_states
    if missing_states:
        raise ValueError(f"Missing expected states: {missing_states}")

    # Validate zero nulls
    null_counts = df[ALL_MODEL_FEATURES].isna().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Features contain null values:\n{null_counts[null_counts > 0]}")

    return df


def build_multi_horizon_table(df: pd.DataFrame, max_horizon: int = 6) -> pd.DataFrame:
    """
    Build multi-horizon training table for direct and recursive forecasting.
    Each row represents (State, Origin Month t, Horizon h) with:
      - target_buy  = Buy_Price(t+h) - Buy_Price(t)
      - target_sell = Sell_Price(t+h) - Sell_Price(t)
      - all features strictly dated <= t (zero data leakage)
    """
    records = []
    for st in STATES:
        st_df = df[df["State"] == st].sort_values("Month_Start_Date").copy().reset_index(drop=True)
        n_rows = len(st_df)

        for i in range(n_rows):
            origin_row = st_df.iloc[i]
            origin_date = origin_row["Month_Start_Date"]
            p_t = origin_row["Buy_Price_INR_kg"]
            s_t = origin_row["Sell_Price_INR_kg"]

            for h in range(1, max_horizon + 1):
                target_idx = i + h
                if target_idx < n_rows:
                    target_row = st_df.iloc[target_idx]
                    target_date = target_row["Month_Start_Date"]
                    p_target = target_row["Buy_Price_INR_kg"]
                    s_target = target_row["Sell_Price_INR_kg"]
                    actual_buy_change = p_target - p_t
                    actual_sell_change = s_target - s_t
                else:
                    target_date = origin_date + pd.DateOffset(months=h)
                    p_target = np.nan
                    s_target = np.nan
                    actual_buy_change = np.nan
                    actual_sell_change = np.nan

                row_dict = {
                    "State": st,
                    "origin": origin_date,
                    "horizon": h,
                    "target_month": target_date,
                    "rate_t": p_t,
                    "rate_actual": p_target,
                    "target_buy": actual_buy_change,
                    "sell_rate_t": s_t,
                    "sell_rate_actual": s_target,
                    "target_sell": actual_sell_change,
                    "weight_tonnes": STATE_WEIGHTS.get(st, 50.0),
                }

                # Features strictly dated at origin t
                for feat in ALL_MODEL_FEATURES:
                    row_dict[feat] = origin_row[feat]

                records.append(row_dict)

    table = pd.DataFrame(records)
    table.sort_values(by=["origin", "horizon", "State"], inplace=True)
    table.reset_index(drop=True, inplace=True)
    return table


def assert_no_leakage(train_df: pd.DataFrame, origin: pd.Timestamp) -> None:
    """Strictly assert that no training target month exceeds the forecast origin."""
    if len(train_df) == 0:
        return
    max_target = train_df["target_month"].max()
    if max_target > origin:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED: Training table reaches {max_target:%Y-%m-%d} "
            f"which is strictly after forecast origin {origin:%Y-%m-%d}!"
        )


if __name__ == "__main__":
    print("=== TESTING STEP 1 PREPROCESSING PIPELINE ===")
    df_raw = load_clean_data()
    print(f"Loaded {len(df_raw)} actual transaction rows across {df_raw['State'].nunique()} states.")
    print("States row distribution:")
    print(df_raw["State"].value_counts())

    table = build_multi_horizon_table(df_raw, max_horizon=6)
    print(f"\nConstructed Multi-Horizon Table: {len(table)} total records across horizons 1 to 6.")
    known_targets = table[table["target_buy"].notna()]
    print(f"Scorable rows with realized actuals: {len(known_targets)}")
    print(f"Total features tracked per row: {len(ALL_MODEL_FEATURES)}")
