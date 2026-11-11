"""
shock_engine.py - What-If Market Scenario Simulation Engine for Sugar Forecasting
==================================================================================
Simulates how fundamental supply, government policy, and agro-climatic shocks
move the quantitative sugar buying and selling price forecasts.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import step1_preprocess as s1
import step3_final_forecast as s3

warnings.filterwarnings("ignore")

# Fundamental Sugar Market Drivers
SUGAR_DRIVERS: Dict[str, Tuple[str, str, str]] = {
    "cane_yield": ("Sugarcane Yield / Harvest Output", "pct", "%"),
    "ethanol_ebp": ("Ethanol Diversion Quota (EBP Sacrifice)", "pct", "%"),
    "sugar_msp": ("Statutory Minimum Selling Price (MSP)", "abs", "INR/kg"),
    "recovery_rate": ("Mill Sucrose Recovery Rate", "abs", "% points"),
    "temp_heat": ("Crushing Season Temperature", "abs", "deg C"),
    "monsoon_rain": ("Monsoon Crop Rainfall", "pct", "%"),
}


class SugarShockEngine:
    """Simulation engine that fits baseline models and evaluates shocked scenarios."""

    def __init__(self):
        self.df_raw = s1.load_clean_data()
        self.table = s1.build_multi_horizon_table(self.df_raw, max_horizon=6)
        self.states = s1.STATES
        self.weights = s1.STATE_WEIGHTS
        self.origin = self.df_raw["Month_Start_Date"].max()
        self.horizon_dates = pd.date_range(self.origin + pd.DateOffset(months=1), periods=6, freq="MS")

        # Fit Baseline Models
        train_data = self.table[self.table["target_buy"].notna()].copy()
        self.X_train = train_data[s1.ALL_MODEL_FEATURES]
        self.y_buy = train_data["target_buy"]
        self.y_sell = train_data["target_sell"]

        self.model_buy = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, min_child_weight=2,
                                      subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                                      random_state=42, n_jobs=1)
        self.model_buy.fit(self.X_train, self.y_buy)

        self.model_sell = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, min_child_weight=2,
                                       subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                                       random_state=42, n_jobs=1)
        self.model_sell.fit(self.X_train, self.y_sell)

    def forecast_scenario(
        self,
        shocks: Dict[str, float] | None = None,
        states_selected: List[str] | None = None
    ) -> pd.DataFrame:
        """
        Run forward 6-month simulation under specified shocks.
        Returns long dataframe with Baseline, Scenario, and Delta across horizons.
        """
        shocks = shocks or {}
        sel_states = states_selected or self.states

        records = []
        for st in sel_states:
            st_last = self.df_raw[self.df_raw["State"] == st].sort_values("Month_Start_Date").iloc[-1]
            last_buy = float(st_last["Buy_Price_INR_kg"])
            last_sell = float(st_last["Sell_Price_INR_kg"])
            weight = float(self.weights.get(st, 50.0))

            base_feats = st_last[s1.ALL_MODEL_FEATURES].to_dict()
            scen_feats = base_feats.copy()

            # Apply Shocks to Features
            if "cane_yield" in shocks and shocks["cane_yield"] != 0:
                mult = 1.0 + (shocks["cane_yield"] / 100.0)
                scen_feats["Sugarcane_Yield_Kg_Per_Ha"] *= mult
                scen_feats["Sugarcane_Production_Lakh_Tonnes"] *= mult
                scen_feats["Monthly_Sugar_Produced_Lakh_Tonnes"] *= mult

            if "ethanol_ebp" in shocks and shocks["ethanol_ebp"] != 0:
                # Higher ethanol diversion lowers net free-sale sugar production
                scen_feats["Monthly_Sugar_Produced_Lakh_Tonnes"] *= (1.0 - (shocks["ethanol_ebp"] / 100.0) * 0.35)

            if "recovery_rate" in shocks and shocks["recovery_rate"] != 0:
                scen_feats["Sugar_Recovery_Rate_Pct"] = max(8.0, scen_feats["Sugar_Recovery_Rate_Pct"] + shocks["recovery_rate"])

            if "temp_heat" in shocks and shocks["temp_heat"] != 0:
                scen_feats["Avg_Temperature_C"] += shocks["temp_heat"]
                scen_feats["Max_Temperature_C"] += shocks["temp_heat"]
                scen_feats["Temp_Range_C"] += abs(shocks["temp_heat"]) * 0.5
                if scen_feats["Max_Temperature_C"] >= 40.0:
                    scen_feats["Heat_Stress_Flag"] = 1

            if "monsoon_rain" in shocks and shocks["monsoon_rain"] != 0:
                mult_rain = 1.0 + (shocks["monsoon_rain"] / 100.0)
                scen_feats["Total_Precipitation_mm"] *= mult_rain
                scen_feats["Rainfall_3M_Cum"] *= mult_rain

            # Policy MSP shock directly impacts floor buying price
            msp_shock = float(shocks.get("sugar_msp", 0.0))

            for h in range(1, 7):
                target_dt = self.origin + pd.DateOffset(months=h)
                m_int = target_dt.month

                # Update target seasonality
                for f_dict in (base_feats, scen_feats):
                    f_dict["Month_sin"] = np.sin(2 * np.pi * m_int / 12)
                    f_dict["Month_cos"] = np.cos(2 * np.pi * m_int / 12)
                    f_dict["Crushing_Season_Flag"] = 1 if m_int in [10, 11, 12, 1, 2, 3, 4] else 0

                # Predict Baseline
                X_base = pd.DataFrame([base_feats])[s1.ALL_MODEL_FEATURES]
                b_buy_chg = float(self.model_buy.predict(X_base)[0])
                b_sell_chg = float(self.model_sell.predict(X_base)[0])
                base_buy_p = round(last_buy + b_buy_chg, 2)
                base_sell_p = round(last_sell + b_sell_chg, 2)

                # Predict Scenario
                X_scen = pd.DataFrame([scen_feats])[s1.ALL_MODEL_FEATURES]
                s_buy_chg = float(self.model_buy.predict(X_scen)[0])
                s_sell_chg = float(self.model_sell.predict(X_scen)[0])

                # Incorporate policy MSP shift if present
                scen_buy_p = round(last_buy + s_buy_chg + msp_shock, 2)
                scen_sell_p = round(last_sell + s_sell_chg + msp_shock * 1.05, 2)

                records.append({
                    "State": st,
                    "Horizon": h,
                    "Horizon_Code": f"N+{h}",
                    "Month": target_dt.strftime("%b %Y"),
                    "Target_Date": target_dt,
                    "Weight": weight,
                    "Baseline_Buy": base_buy_p,
                    "Scenario_Buy": scen_buy_p,
                    "Delta_Buy": round(scen_buy_p - base_buy_p, 2),
                    "Baseline_Sell": base_sell_p,
                    "Scenario_Sell": scen_sell_p,
                    "Delta_Sell": round(scen_sell_p - base_sell_p, 2),
                    "Baseline_Margin": round(base_sell_p - base_buy_p, 2),
                    "Scenario_Margin": round(scen_sell_p - scen_buy_p, 2),
                })

        return pd.DataFrame(records)

    def aggregate_scenario(self, df_scen: pd.DataFrame) -> pd.DataFrame:
        """Volume-weighted national aggregate across horizons."""
        agg_rows = []
        for (h, h_code, mth, dt), grp in df_scen.groupby(["Horizon", "Horizon_Code", "Month", "Target_Date"]):
            w = grp["Weight"]
            tot_w = w.sum()

            b_buy = round(float((grp["Baseline_Buy"] * w).sum() / tot_w), 2)
            s_buy = round(float((grp["Scenario_Buy"] * w).sum() / tot_w), 2)
            b_sell = round(float((grp["Baseline_Sell"] * w).sum() / tot_w), 2)
            s_sell = round(float((grp["Scenario_Sell"] * w).sum() / tot_w), 2)

            agg_rows.append({
                "Horizon": h,
                "Horizon_Code": h_code,
                "Month": mth,
                "Target_Date": dt,
                "Baseline_Buy": b_buy,
                "Scenario_Buy": s_buy,
                "Delta_Buy": round(s_buy - b_buy, 2),
                "Baseline_Sell": b_sell,
                "Scenario_Sell": s_sell,
                "Delta_Sell": round(s_sell - b_sell, 2),
                "Baseline_Margin": round(b_sell - b_buy, 2),
                "Scenario_Margin": round(s_sell - s_buy, 2),
                "Delta_Margin": round((s_sell - s_buy) - (b_sell - b_buy), 2)
            })

        return pd.DataFrame(agg_rows).sort_values("Horizon").reset_index(drop=True)

    def run_tornado_analysis(self, horizon: int = 3) -> pd.DataFrame:
        """Push each driver up and down independently to measure price swing sensitivity."""
        tornado_rows = []

        test_shocks = {
            "cane_yield": 15.0,        # +15% / -15%
            "ethanol_ebp": 20.0,       # +20% / -20%
            "sugar_msp": 3.0,          # +3 INR / -3 INR
            "recovery_rate": 0.5,      # +0.5% / -0.5% recovery
            "temp_heat": 2.0,          # +2 deg C / -2 deg C
            "monsoon_rain": 25.0,      # +25% / -25%
        }

        base_agg = self.aggregate_scenario(self.forecast_scenario({}))
        base_h = base_agg[base_agg["Horizon"] == horizon].iloc[0]["Baseline_Buy"]

        for k, (label, kind, unit) in SUGAR_DRIVERS.items():
            size = test_shocks[k]

            # Positive Shock
            up_agg = self.aggregate_scenario(self.forecast_scenario({k: size}))
            up_val = up_agg[up_agg["Horizon"] == horizon].iloc[0]["Scenario_Buy"]
            up_effect = round(up_val - base_h, 2)

            # Negative Shock
            dn_agg = self.aggregate_scenario(self.forecast_scenario({k: -size}))
            dn_val = dn_agg[dn_agg["Horizon"] == horizon].iloc[0]["Scenario_Buy"]
            dn_effect = round(dn_val - base_h, 2)

            swing = round(abs(up_effect - dn_effect), 2)

            tornado_rows.append({
                "Driver": label,
                "Key": k,
                "Shock_Size": f"±{size:g} {unit}",
                "Rise_Effect": up_effect,
                "Fall_Effect": dn_effect,
                "Swing_INR_kg": swing
            })

        out = pd.DataFrame(tornado_rows).sort_values("Swing_INR_kg", ascending=False).reset_index(drop=True)
        return out
