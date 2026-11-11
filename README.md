# Staples Commodity Forecasting – Indian Sugar

This project implements an end-to-end Machine Learning forecasting pipeline and interactive Streamlit application for Indian Sugar procurement (buying) and wholesale (selling) prices across the Top 5 producing and consuming states: **Karnataka, Uttar Pradesh, Gujarat, Maharashtra, and West Bengal**.

The methodology is adapted from the **Buffalo Milk Price Forecasting** architecture (`RecursiveX XGBoost Regressor` + `Direct Ridge` + 6-batch Multi-Holdout Backtest), completely independent of any dairy-specific logic.

---

## 1. Project Directory Structure

```
staples_forecasting/
├── app.py                             # Streamlit interactive forecasting & validation dashboard
├── app.yaml                           # Databricks Apps deployment specification
├── requirements.txt                   # Production dependencies
├── README.md                          # Project documentation and execution instructions
├── speculation_loader.py              # External sugar trade expectations & validation engine
├── shock_engine.py                    # What-if market scenario simulation engine
├── step1_preprocess.py                # Data loading, validation, and multi-horizon table builder
├── step2_backtest.py                  # 6-data-point multi-holdout backtesting pipeline
├── step3_final_forecast.py            # Final 6-month forward forecast (Dec 2026 - May 2027)
├── external_market_speculations.json  # Verified external market intelligence dossier
├── data/
│   ├── Sugar_Top5_States_Clean_Input_Dataset.csv
│   └── Sugar_Top5_States_Clean_Input_Dataset.xlsx
└── outputs/
    ├── Sugar_MultiHoldout_Report_6Batches.xlsx
    ├── Sugar_MultiHoldout_Report_6Batches.csv
    ├── Sugar_Final_Forecast_Dec2026_May2027.xlsx
    ├── Sugar_Final_Forecast_Dec2026_May2027.csv
    ├── holdout_detail.csv
    └── backtest_meta.json
```

---

## 2. Key Methodological Features

1. **Dual Price Forecasting Models:**
   - Independent models for **Buying Price** (mill procurement rate) and **Selling Price** (wholesale realization).
   - Commercial gross margins ($\text{INR/kg}$ and $\%$) derived without artificial inflation.
2. **Buffalo RecursiveX XGBoost Architecture:**
   - 1-step autoregressive model fed back recursively to project multi-step ahead horizons ($N+1$ to $N+6$).
   - Regularized gradient boosting (`max_depth=3`, `learning_rate=0.05`, `colsample_bytree=0.8`, `reg_lambda=2.0`).
   - Penalized `RidgeCV` multi-horizon direct models for linear stability and ensemble blending.
3. **Six-Data-Point Multi-Holdout Validation:**
   - Evaluated across 6 historical origins spaced 2 months apart over the past year:
     `2025-07-01`, `2025-09-01`, `2025-11-01`, `2026-01-01`, `2026-03-01`, `2026-05-01`.
   - Strict no-leakage protocol: for any origin $T$, algorithms train only on transactions where $t \le T$.
   - Rigorously checks benchmark error bounds:
     * $N+1$ MAE $\le 2.0$ ₹/kg
     * Q1 MAE $\le 5.0$ ₹/kg
     * Q2 MAE $\le 8.0$ ₹/kg
4. **Authentic External Market Outlook Integration:**
   - Curated market expectations from official industry sources: **ISMA** (Indian Sugar & Bio-energy Manufacturers Association), **NFCSF** (National Federation of Cooperative Sugar Factories), **AgriWatch**, **ChiniMandi**, and **The Hindu BusinessLine**.
   - Calculates statistical agreement metrics (Directional Agreement %, Pearson correlation $r$) against quantitative model predictions.
5. **No Crash-Period Analysis:**
   - Crash analysis and crash-related calculations have been completely omitted per requirements.

---

## 3. Step-by-Step Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Data Preprocessing & Validation
```bash
python step1_preprocess.py
```

### Step 3: Execute 6-Batch Multi-Holdout Backtest
```bash
python step2_backtest.py
```
*Generates `Sugar_MultiHoldout_Report_6Batches.xlsx` and `backtest_meta.json`.*

### Step 4: Generate Final 6-Month Forward Forecast (Dec 2026 – May 2027)
```bash
python step3_final_forecast.py
```
*Generates `Sugar_Final_Forecast_Dec2026_May2027.xlsx` with state and portfolio forecasts + 80% confidence bands.*

### Step 5: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 4. Databricks Apps Deployment

This repository includes a native `app.yaml` manifest configured for **Databricks Apps**:
```yaml
command: [
  "streamlit",
  "run",
  "app.py"
]
```

To deploy on Databricks Apps:
1. Sync or upload the `staples_forecasting` folder to your Databricks workspace.
2. Create a new Databricks App selecting Python runtime with Streamlit.
3. Point the application source to `staples_forecasting/app.py`.
