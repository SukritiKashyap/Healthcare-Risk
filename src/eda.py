"""
eda.py
Exploratory Data Analysis: missing values, outliers, distributions,
correlations and business insights. Saves figures to reports/.
Run from project root: python src/eda.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)


def main():
    df = pd.read_csv("data/healthcare_claims.csv")
    print(f"Shape: {df.shape}")

    # --- 1. Missing values -----------------------------------------------
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print("\n=== MISSING VALUES ===")
    print((miss / len(df) * 100).round(2).astype(str) + "%")

    # --- 2. Outlier analysis (IQR) for key monetary fields ----------------
    print("\n=== OUTLIERS (IQR method) ===")
    for col in ["Previous_Year_Claim_Amount", "Pharmacy_Spend",
                "Average_Claim_Size"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        hi = q3 + 1.5 * iqr
        n_out = (df[col] > hi).sum()
        print(f"{col:30s} upper_fence={hi:>12,.0f}  outliers={n_out:>4} "
              f"({n_out/len(df):.1%})")

    # --- 3. Distribution / summary ---------------------------------------
    print("\n=== NUMERIC SUMMARY ===")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]].round(1))

    # --- 4. Correlation with target --------------------------------------
    num = df.select_dtypes(include=np.number)
    corr_t = num.corr()["High_Cost_Member"].drop("High_Cost_Member")
    print("\n=== CORRELATION WITH High_Cost_Member ===")
    print(corr_t.sort_values(ascending=False).round(3))

    # --- 5. Business insights --------------------------------------------
    print("\n=== BUSINESS INSIGHTS ===")
    hc = df[df.High_Cost_Member == 1]
    base = df[df.High_Cost_Member == 0]
    print(f"High-cost members avg prior claim: {hc.Previous_Year_Claim_Amount.mean():,.0f} "
          f"vs {base.Previous_Year_Claim_Amount.mean():,.0f} (others)")
    print(f"High-cost avg chronic conditions:  {hc.Chronic_Conditions_Count.mean():.2f} "
          f"vs {base.Chronic_Conditions_Count.mean():.2f}")
    print(f"High-cost avg ER visits:           {hc.Emergency_Visits.mean():.2f} "
          f"vs {base.Emergency_Visits.mean():.2f}")
    rate_by_heart = df.groupby("Heart_Disease").High_Cost_Member.mean()
    print(f"High-cost rate | heart disease=1:  {rate_by_heart.get(1,0):.1%} "
          f"vs no heart disease: {rate_by_heart.get(0,0):.1%}")

    # --- Plots ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    df["Previous_Year_Claim_Amount"].clip(upper=df.Previous_Year_Claim_Amount.quantile(0.99)).hist(
        bins=50, ax=ax[0, 0]); ax[0, 0].set_title("Prior Year Claim Amount")
    df["Age"].hist(bins=40, ax=ax[0, 1]); ax[0, 1].set_title("Age")
    df.groupby("Chronic_Conditions_Count").High_Cost_Member.mean().plot(
        kind="bar", ax=ax[1, 0]); ax[1, 0].set_title("High-Cost Rate by Chronic Count")
    df.boxplot(column="Pharmacy_Spend", by="High_Cost_Member", ax=ax[1, 1])
    ax[1, 1].set_title("Pharmacy Spend by Target"); ax[1, 1].set_ylim(0, 40000)
    plt.suptitle(""); plt.tight_layout()
    plt.savefig("reports/eda_distributions.png", dpi=120)
    plt.close()

    plt.figure(figsize=(11, 9))
    c = num.corr()
    plt.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046)
    plt.xticks(range(len(c)), c.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(c)), c.columns, fontsize=7)
    plt.title("Correlation Matrix"); plt.tight_layout()
    plt.savefig("reports/eda_correlation.png", dpi=120)
    plt.close()
    print("\nFigures saved to reports/.")


if __name__ == "__main__":
    main()
