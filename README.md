# Healthcare Cost Risk Prediction & Explainable Member Risk Scoring

Predicts which health-insurance members are likely to become **high-cost claimants
in the next year** and assigns each member an **explainable risk score** (Low →
Critical) that care-management, underwriting and finance teams can act on.

Built to mirror how a risk/analytics team inside a health insurer actually works:
cost-persistence modelling, a defensible scoring framework, SHAP-based
transparency, an analyst SQL layer, and a Power BI dashboard spec for the business.

---

## 1. Business Problem

In most health-insurance books, a small fraction of members drives the majority
of spend. Acting *after* a member becomes high-cost is too late — the claims are
already incurred. The business needs to **identify rising-risk members early**,
**explain why** they are flagged (regulators, clinicians and underwriters will not
trust a black box), and **prioritise finite care-management capacity** where it
yields the most savings.

**Goal:** Build a model that flags the top ~20% of next-year spenders and converts
its output into a member-level risk score with clear, auditable drivers.

## 2. Data

Synthetic but internally consistent member-level dataset (`data/healthcare_claims.csv`),
**12,000 members**. The target (`High_Cost_Member`) is defined on a *separately
simulated next-year spend* so the model predicts the future rather than a leaked
copy of present cost. Realistic missingness is injected into BMI, Pharmacy_Spend
and Smoking_Status.

| Group | Fields |
|---|---|
| Demographics | Age, Gender, Region, BMI, Smoking_Status |
| Clinical | Chronic_Conditions_Count, Diabetes, Hypertension, Heart_Disease |
| Utilization | Hospital_Visits, Emergency_Visits, Number_of_Claims |
| Financial | Pharmacy_Spend, Previous_Year_Claim_Amount, Average_Claim_Size |
| Target | High_Cost_Member (1 = top 20% future spenders) |

## 3. Methodology

1. **Imputation** — median (numeric) / mode (categorical).
2. **Feature engineering** (`src/features.py`): Healthcare Utilization Score,
   Clinical Risk Index, Chronic Disease Burden Score, claim frequency metrics
   (claims-per-visit, ER-to-hospital ratio), and cost ratios
   (pharmacy-to-claim, cost-per-condition, avg-to-total). All leakage-safe.
3. **Encoding & split** — one-hot for Gender/Region; stratified 75/25 split.
4. **Models** — Logistic Regression (interpretable baseline, scaled,
   class-weighted), Random Forest, XGBoost (with `scale_pos_weight` for imbalance).
5. **Evaluation** — ROC-AUC, Precision, Recall, F1, Confusion Matrix.
6. **Explainability** — SHAP global importance + per-member explanations.
7. **Scoring framework** — predicted probability → Low / Medium / High / Critical.

## 4. Key Findings

- All three models separate high-cost members strongly (**ROC-AUC ≈ 0.92**).
  The pipeline is tuned for **high recall on high-cost members (~0.83)** —
  in care management, missing a future high-cost member costs far more than a
  false positive.
- **Top drivers (SHAP):** Previous-Year Claim Amount, Utilization Score, Risk
  Index, Age, Pharmacy-to-Claim ratio, Chronic Conditions Count — i.e. cost
  *persistence* plus clinical burden, exactly as actuarial intuition predicts.
- **Cost concentration:** members in the Critical/High bands hold a
  disproportionate share of expected spend, making them the clear ROI target
  for outreach.
- **Clinical signal:** members with heart disease show a high-cost rate of ~59%
  vs ~13% without — a strong, explainable underwriting/care flag.

## 5. Recommendations

1. **Operationalise the score** as a monthly batch: feed Critical/High bands to
   care management for proactive outreach.
2. **Tier interventions** — Critical: case manager; High: nurse call + pharmacy
   review; Medium: digital engagement; Low: monitor.
3. **Use SHAP explanations** as care-manager talking points and to satisfy
   audit/regulatory "right to explanation" expectations.
4. **Monitor drift** — re-score quarterly; track band migration as an early
   warning of deteriorating cohorts.

## 6. How to Run

```bash
pip install -r requirements.txt        # pandas numpy scikit-learn xgboost shap matplotlib
python src/generate_data.py            # -> data/healthcare_claims.csv
python src/eda.py                      # -> reports/ figures + console insights
python src/train.py                    # -> models, SHAP, member_risk_scores.csv
python src/explain_member.py M100042   # per-member explanation
```
