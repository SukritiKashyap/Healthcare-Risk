-- ============================================================================
-- analyst_queries.sql
-- Healthcare Member Risk -- analyst-facing SQL
--
-- Assumes two tables loaded from the pipeline output:
--   member_claims(member_id, age, gender, region, bmi,
--                 chronic_conditions_count, hospital_visits, emergency_visits,
--                 pharmacy_spend, previous_year_claim_amount, number_of_claims,
--                 average_claim_size, smoking_status, diabetes, hypertension,
--                 heart_disease, high_cost_member)
--   member_risk_scores(member_id, risk_probability, risk_score, risk_band)
-- Dialect: ANSI SQL (tested against standard syntax; minor tweaks for T-SQL/
-- BigQuery noted where relevant).
-- ============================================================================


-- ----------------------------------------------------------------------------
-- 1. IDENTIFY HIGH-RISK MEMBERS  (for care-management outreach lists)
-- ----------------------------------------------------------------------------
SELECT
    s.member_id,
    s.risk_score,
    s.risk_band,
    c.age,
    c.region,
    c.chronic_conditions_count,
    c.previous_year_claim_amount,
    c.emergency_visits
FROM member_risk_scores s
JOIN member_claims c ON c.member_id = s.member_id
WHERE s.risk_band IN ('High Risk', 'Critical Risk')
ORDER BY s.risk_score DESC;


-- Critical-risk members who are NOT yet flagged as high-cost (emerging risk):
SELECT s.member_id, s.risk_score, c.chronic_conditions_count, c.emergency_visits
FROM member_risk_scores s
JOIN member_claims c ON c.member_id = s.member_id
WHERE s.risk_band = 'Critical Risk'
  AND c.high_cost_member = 0
ORDER BY s.risk_score DESC;


-- ----------------------------------------------------------------------------
-- 2. CLAIM TRENDS  (cost concentration & cohort comparison)
-- ----------------------------------------------------------------------------
-- Cost concentration: what share of total prior-year spend comes from each band?
SELECT
    s.risk_band,
    COUNT(*)                                            AS members,
    SUM(c.previous_year_claim_amount)                   AS total_claims,
    ROUND(AVG(c.previous_year_claim_amount), 0)         AS avg_claim,
    ROUND(100.0 * SUM(c.previous_year_claim_amount)
          / SUM(SUM(c.previous_year_claim_amount)) OVER (), 1) AS pct_of_total_spend
FROM member_risk_scores s
JOIN member_claims c ON c.member_id = s.member_id
GROUP BY s.risk_band
ORDER BY total_claims DESC;


-- Average claim size & utilization by chronic-condition burden:
SELECT
    chronic_conditions_count,
    COUNT(*)                              AS members,
    ROUND(AVG(previous_year_claim_amount),0) AS avg_annual_claim,
    ROUND(AVG(hospital_visits), 2)        AS avg_hospital_visits,
    ROUND(AVG(emergency_visits), 2)       AS avg_er_visits,
    ROUND(AVG(high_cost_member) * 100, 1) AS high_cost_rate_pct
FROM member_claims
GROUP BY chronic_conditions_count
ORDER BY chronic_conditions_count;


-- ----------------------------------------------------------------------------
-- 3. MANAGEMENT REPORT  (region x risk -- exec summary feed)
-- ----------------------------------------------------------------------------
SELECT
    c.region,
    COUNT(*)                                                   AS total_members,
    SUM(CASE WHEN s.risk_band = 'Critical Risk' THEN 1 ELSE 0 END) AS critical_members,
    ROUND(100.0 * SUM(CASE WHEN s.risk_band IN ('High Risk','Critical Risk')
          THEN 1 ELSE 0 END) / COUNT(*), 1)                    AS pct_high_or_critical,
    ROUND(AVG(s.risk_score), 1)                                AS avg_risk_score,
    ROUND(SUM(c.previous_year_claim_amount), 0)                AS region_total_claims,
    ROUND(AVG(c.previous_year_claim_amount), 0)                AS region_avg_claim
FROM member_risk_scores s
JOIN member_claims c ON c.member_id = s.member_id
GROUP BY c.region
ORDER BY pct_high_or_critical DESC;


-- Top 100 members to prioritise for care management this quarter:
SELECT s.member_id, s.risk_score, s.risk_band,
       c.previous_year_claim_amount, c.chronic_conditions_count
FROM member_risk_scores s
JOIN member_claims c ON c.member_id = s.member_id
ORDER BY s.risk_score DESC
LIMIT 100;          -- T-SQL: use SELECT TOP 100 ...
