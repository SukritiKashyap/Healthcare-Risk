"""
generate_data.py
Synthetic healthcare claims data generator for the Member Risk Prediction project.

Produces a realistic, internally-consistent member-level dataset where future
high-cost status is driven by a believable mix of clinical, utilization and
demographic signals (plus noise). The target is defined on a *separately
simulated* next-year spend so the model is predicting the future, not a leaked
copy of the present.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 12_000  # >10k records


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


def generate(n: int = N) -> pd.DataFrame:
    # --- Demographics -----------------------------------------------------
    age = _clip(RNG.normal(45, 16, n), 18, 90).round().astype(int)
    gender = RNG.choice(["Male", "Female"], n, p=[0.49, 0.51])
    region = RNG.choice(
        ["North", "South", "East", "West", "Central"],
        n, p=[0.22, 0.26, 0.18, 0.20, 0.14],
    )

    # BMI rises mildly with age
    bmi = _clip(RNG.normal(26 + (age - 45) * 0.03, 4.5, n), 15, 50).round(1)

    smoking = RNG.choice(["Never", "Former", "Current"], n, p=[0.58, 0.27, 0.15])

    # --- Clinical conditions (age / bmi / smoking driven) -----------------
    p_diab = _clip(0.04 + (age - 18) * 0.004 + (bmi - 25) * 0.012
                   + (smoking == "Current") * 0.05, 0.01, 0.85)
    diabetes = RNG.binomial(1, p_diab)

    p_htn = _clip(0.05 + (age - 18) * 0.006 + (bmi - 25) * 0.010
                  + (smoking == "Current") * 0.06, 0.01, 0.9)
    hypertension = RNG.binomial(1, p_htn)

    p_heart = _clip(0.01 + (age - 18) * 0.004 + diabetes * 0.08
                    + hypertension * 0.07 + (smoking == "Current") * 0.06, 0.005, 0.8)
    heart_disease = RNG.binomial(1, p_heart)

    chronic_count = (diabetes + hypertension + heart_disease
                     + RNG.binomial(2, 0.12, n))  # +other chronic conditions

    # --- Utilization ------------------------------------------------------
    visit_lambda = 0.6 + chronic_count * 0.9 + (age > 60) * 0.8
    hospital_visits = RNG.poisson(visit_lambda)
    emergency_visits = RNG.poisson(0.2 + chronic_count * 0.35
                                   + (smoking == "Current") * 0.2)

    number_of_claims = RNG.poisson(2 + chronic_count * 1.8 + hospital_visits * 1.2) + 1

    pharmacy_spend = _clip(
        RNG.gamma(2.0, 1500) * (1 + chronic_count * 0.5) / 2, 0, 80_000
    ).round(0)

    avg_claim_size = _clip(
        RNG.gamma(2.0, 4000) * (1 + chronic_count * 0.4
                                + heart_disease * 0.6) / 2, 200, 120_000
    ).round(0)

    prev_year_claim = (number_of_claims * avg_claim_size
                       * RNG.uniform(0.8, 1.2, n)).round(0)
    prev_year_claim = _clip(prev_year_claim, 0, 1_500_000)

    # --- Next-year spend (the thing we actually predict) ------------------
    # Persistence of cost + acceleration from risk factors + shocks
    latent = (
        0.55 * np.log1p(prev_year_claim)
        + 0.20 * np.log1p(pharmacy_spend)
        + 0.35 * chronic_count
        + 0.30 * emergency_visits
        + 0.25 * hospital_visits
        + 0.015 * age
        + 0.40 * heart_disease
        + 0.20 * diabetes
        + 0.15 * (smoking == "Current")
        + 0.02 * (bmi - 25)
        + RNG.normal(0, 1.1, n)  # irreducible noise
    )
    next_year_spend = np.expm1(latent / 1.5) * 300
    threshold = np.quantile(next_year_spend, 0.80)
    high_cost = (next_year_spend >= threshold).astype(int)

    df = pd.DataFrame({
        "Member_ID": [f"M{100000+i}" for i in range(n)],
        "Age": age,
        "Gender": gender,
        "Region": region,
        "BMI": bmi,
        "Chronic_Conditions_Count": chronic_count,
        "Hospital_Visits": hospital_visits,
        "Emergency_Visits": emergency_visits,
        "Pharmacy_Spend": pharmacy_spend,
        "Previous_Year_Claim_Amount": prev_year_claim,
        "Number_of_Claims": number_of_claims,
        "Average_Claim_Size": avg_claim_size,
        "Smoking_Status": smoking,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Heart_Disease": heart_disease,
        "High_Cost_Member": high_cost,
    })

    # --- Inject realistic missingness (MAR) -------------------------------
    for col, frac in [("BMI", 0.04), ("Pharmacy_Spend", 0.03),
                      ("Smoking_Status", 0.02)]:
        idx = RNG.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate()
    out = "data/healthcare_claims.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")
    print(f"High-cost prevalence: {df.High_Cost_Member.mean():.1%}")
