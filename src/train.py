"""
train.py
End-to-end training pipeline:
  load -> impute -> feature engineer -> encode -> split
  -> train (LogReg / RandomForest / XGBoost) -> evaluate -> SHAP -> risk scores

Run from project root:  python src/train.py
Outputs metrics, a comparison table, SHAP plots and a scored member file.
"""

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
import shap

import features as F

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
def load_and_prepare(path="data/healthcare_claims.csv"):
    df = pd.read_csv(path)

    # Impute missing: numeric -> median, categorical -> mode
    df["BMI"] = df["BMI"].fillna(df["BMI"].median())
    df["Pharmacy_Spend"] = df["Pharmacy_Spend"].fillna(
        df["Pharmacy_Spend"].median())
    df["Smoking_Status"] = df["Smoking_Status"].fillna(
        df["Smoking_Status"].mode()[0])

    df = F.add_features(df)

    # One-hot encode categoricals
    df["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    for r in ["North", "South", "East", "West", "Central"]:
        df[f"Region_{r}"] = (df["Region"] == r).astype(int)

    return df


def evaluate(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "ROC_AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
    }


def risk_band(p):
    if p < 0.20:
        return "Low Risk"
    if p < 0.50:
        return "Medium Risk"
    if p < 0.80:
        return "High Risk"
    return "Critical Risk"


# --------------------------------------------------------------------------- #
def main():
    df = load_and_prepare()
    X = df[F.FEATURE_COLS]
    y = df["High_Cost_Member"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # Scale for the linear model
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results, fitted = [], {}

    # --- Logistic Regression ---------------------------------------------
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    p = lr.predict_proba(X_test_s)[:, 1]
    results.append(evaluate("Logistic Regression", y_test,
                            (p >= 0.5).astype(int), p))
    fitted["Logistic Regression"] = lr

    # --- Random Forest ----------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    p = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test,
                            (p >= 0.5).astype(int), p))
    fitted["Random Forest"] = rf

    # --- XGBoost ----------------------------------------------------------
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test,
                            (p_xgb >= 0.5).astype(int), p_xgb))
    fitted["XGBoost"] = xgb

    # --- Comparison table -------------------------------------------------
    comp = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    print("\n=== MODEL COMPARISON ===")
    print(comp.to_string(index=False))
    comp.to_csv("reports/model_comparison.csv", index=False)

    best_name = comp.iloc[0]["Model"]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Confusion matrix + report for best
    best_proba = (best.predict_proba(X_test_s if best_name ==
                  "Logistic Regression" else X_test)[:, 1])
    best_pred = (best_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, best_pred)
    print("\nConfusion Matrix [best]:\n", cm)
    print("\n", classification_report(y_test, best_pred,
          target_names=["Not High-Cost", "High-Cost"]))

    # --- SHAP (tree model) -----------------------------------------------
    tree_model = fitted["XGBoost"]
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig("reports/shap_global_importance.png", dpi=130,
                bbox_inches="tight")
    plt.close()

    mean_abs = pd.DataFrame({
        "feature": F.FEATURE_COLS,
        "mean_abs_shap": np.abs(shap_values).mean(0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs.to_csv("reports/shap_feature_importance.csv", index=False)
    print("\n=== TOP 10 SHAP DRIVERS ===")
    print(mean_abs.head(10).to_string(index=False))

    # --- Score the full population ---------------------------------------
    full_proba = tree_model.predict_proba(X)[:, 1]
    scored = df[["Member_ID", "Age", "Previous_Year_Claim_Amount",
                 "Chronic_Conditions_Count", "High_Cost_Member"]].copy()
    scored["Risk_Probability"] = full_proba.round(4)
    scored["Risk_Score"] = (full_proba * 100).round(1)
    scored["Risk_Band"] = [risk_band(p) for p in full_proba]
    scored.to_csv("data/member_risk_scores.csv", index=False)

    print("\n=== RISK BAND DISTRIBUTION ===")
    print(scored["Risk_Band"].value_counts()
          .reindex(["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]))

    # Save metrics json for the report
    with open("reports/metrics.json", "w") as f:
        json.dump({"comparison": results, "best_model": best_name,
                   "confusion_matrix": cm.tolist(),
                   "top_drivers": mean_abs.head(10).to_dict("records")},
                  f, indent=2)
    print("\nArtifacts written to reports/ and data/.")


if __name__ == "__main__":
    main()
