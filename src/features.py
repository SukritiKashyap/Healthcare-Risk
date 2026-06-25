"""
features.py
Feature engineering for the member risk model.

Creates the composite, domain-driven features an actuarial / risk analyst
would actually build: a utilization score, a clinical risk index, claim
frequency/severity metrics, cost ratios and a chronic disease burden score.
All transforms are leakage-safe (no use of next-year spend).
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Healthcare Utilization Score -- weighted touchpoints with the system
    df["Utilization_Score"] = (
        df["Hospital_Visits"] * 3
        + df["Emergency_Visits"] * 5      # ER weighted highest (acute / costly)
        + df["Number_of_Claims"] * 1
    )

    # 2. Chronic Disease Burden Score -- weighted clinical load
    df["Chronic_Burden_Score"] = (
        df["Diabetes"] * 2
        + df["Hypertension"] * 1.5
        + df["Heart_Disease"] * 3
        + df["Chronic_Conditions_Count"] * 1
    )

    # 3. Clinical Risk Index -- demographics + lifestyle + clinical
    smoke_map = {"Never": 0, "Former": 1, "Current": 3}
    df["Smoking_Risk"] = df["Smoking_Status"].map(smoke_map).fillna(0)
    df["Risk_Index"] = (
        (df["Age"] / 10)
        + (df["BMI"] - 25).clip(lower=0) * 0.3
        + df["Smoking_Risk"] * 2
        + df["Chronic_Burden_Score"]
    )

    # 4. Claim frequency / severity metrics
    df["Claims_Per_Visit"] = df["Number_of_Claims"] / (
        df["Hospital_Visits"] + df["Emergency_Visits"] + 1)
    df["ER_to_Hospital_Ratio"] = df["Emergency_Visits"] / (
        df["Hospital_Visits"] + 1)

    # 5. Cost ratios
    df["Pharmacy_to_Claim_Ratio"] = df["Pharmacy_Spend"] / (
        df["Previous_Year_Claim_Amount"] + 1)
    df["Cost_Per_Condition"] = df["Previous_Year_Claim_Amount"] / (
        df["Chronic_Conditions_Count"] + 1)
    df["Avg_to_Total_Claim_Ratio"] = df["Average_Claim_Size"] / (
        df["Previous_Year_Claim_Amount"] + 1)

    return df


FEATURE_COLS = [
    "Age", "BMI", "Chronic_Conditions_Count", "Hospital_Visits",
    "Emergency_Visits", "Pharmacy_Spend", "Previous_Year_Claim_Amount",
    "Number_of_Claims", "Average_Claim_Size", "Diabetes", "Hypertension",
    "Heart_Disease", "Utilization_Score", "Chronic_Burden_Score",
    "Smoking_Risk", "Risk_Index", "Claims_Per_Visit", "ER_to_Hospital_Ratio",
    "Pharmacy_to_Claim_Ratio", "Cost_Per_Condition", "Avg_to_Total_Claim_Ratio",
    "Gender_Male", "Region_North", "Region_South", "Region_East",
    "Region_West", "Region_Central",
]
