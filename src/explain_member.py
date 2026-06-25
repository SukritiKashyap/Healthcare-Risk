"""
explain_member.py
Generate an individual, plain-English risk explanation for a single member
using SHAP. This is what powers the 'Member Drill-down' dashboard tooltip and
care-manager talking points.

Usage:  python src/explain_member.py M100042
"""

import sys
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

import features as F
from train import load_and_prepare, RANDOM_STATE


def build_model(X, y):
    spw = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X, y)
    return model


def explain(member_id: str):
    df = load_and_prepare()
    X = df[F.FEATURE_COLS]
    y = df["High_Cost_Member"]
    model = build_model(X, y)

    row = df.index[df["Member_ID"] == member_id]
    if len(row) == 0:
        print(f"Member {member_id} not found."); return
    i = row[0]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X.iloc[[i]])[0]
    prob = model.predict_proba(X.iloc[[i]])[0, 1]

    contrib = (pd.DataFrame({"feature": F.FEATURE_COLS,
                             "value": X.iloc[i].values,
                             "shap": sv})
               .assign(abs_shap=lambda d: d.shap.abs())
               .sort_values("abs_shap", ascending=False))

    print(f"\nMember {member_id}  |  Risk probability: {prob:.1%}")
    print("Top factors INCREASING risk:")
    for _, r in contrib[contrib.shap > 0].head(5).iterrows():
        print(f"   + {r.feature:28s} (value={r.value:.1f})")
    print("Top factors DECREASING risk:")
    for _, r in contrib[contrib.shap < 0].head(3).iterrows():
        print(f"   - {r.feature:28s} (value={r.value:.1f})")


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else "M100042"
    explain(mid)
