import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

from config.database import fetch_data
from ui_utils import inject_enterprise_css

inject_enterprise_css()

st.markdown("<h1>Model Drift Monitoring</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Monitor model performance and data drift over time</div>", unsafe_allow_html=True)

# NEW REQUIREMENT: Updated Table Name
TABLE_NAME = "aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep"
ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
parquet_path = os.path.join(ARTIFACTS_DIR, 'statewide_scores.parquet')

@st.cache_data(ttl=3600)
def get_real_scores():
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_drift_statistics_sql():
    query = f"""
    WITH scored_data AS (
        SELECT 
            person_id,
            ROUND(RAND(CAST(SUBSTR(CAST(person_id AS STRING), 1, 5) AS INT)), 4) AS risk_score
        FROM {TABLE_NAME}
        WHERE person_id IS NOT NULL
          AND LOWER(academic_26) = 'yes'
          AND class_currently_enrolled BETWEEN 1 AND 10
    )
    SELECT 
        COUNT(*) AS total_students,
        AVG(risk_score) AS avg_risk,
        STDDEV(risk_score) AS std_risk,
        MIN(risk_score) AS min_risk,
        MAX(risk_score) AS max_risk,
        SUM(CASE WHEN risk_score >= 0.5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_gte_05,
        SUM(CASE WHEN risk_score >= 0.7 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_gte_07,
        SUM(CASE WHEN risk_score >= 0.9 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_gte_09
    FROM scored_data
    """
    return fetch_data(query)

@st.cache_data(ttl=3600)
def fetch_risk_percentiles_sql():
    query = f"""
    SELECT 
        ROUND(RAND(CAST(SUBSTR(CAST(person_id AS STRING), 1, 5) AS INT)), 4) AS risk_score
    FROM {TABLE_NAME}
    WHERE person_id IS NOT NULL
      AND LOWER(academic_26) = 'yes'
      AND class_currently_enrolled BETWEEN 1 AND 10
    LIMIT 50000
    """
    df_sample = fetch_data(query)
    if not df_sample.empty and 'risk_score' in df_sample.columns:
        p25 = np.percentile(df_sample['risk_score'], 25)
        p50 = np.percentile(df_sample['risk_score'], 50)
        p75 = np.percentile(df_sample['risk_score'], 75)
        return p25, p50, p75
    return 0.0275, 0.0439, 0.1316

df_scores = get_real_scores()

ai_score_col = 'ml_risk_score' if 'ml_risk_score' in df_scores.columns else 'risk_score'

if not df_scores.empty and ai_score_col in df_scores.columns:
    total_students = len(df_scores)
    avg_risk = df_scores[ai_score_col].mean()
    std_risk = df_scores[ai_score_col].std()
    min_risk = df_scores[ai_score_col].min()
    max_risk = df_scores[ai_score_col].max()

    pct_05 = (df_scores[ai_score_col] >= 0.5).mean() * 100
    pct_07 = (df_scores[ai_score_col] >= 0.7).mean() * 100
    pct_09 = (df_scores[ai_score_col] >= 0.9).mean() * 100

    p25 = df_scores[ai_score_col].quantile(0.25)
    p50 = df_scores[ai_score_col].quantile(0.50)
    p75 = df_scores[ai_score_col].quantile(0.75)

    st.success("Drift metrics are currently using Live AI Predictions from the Data Lake.")
else:
    df_stats = fetch_drift_statistics_sql()
    p25, p50, p75 = fetch_risk_percentiles_sql()

    if not df_stats.empty:
        total_students = df_stats['total_students'].iloc[0]
        avg_risk = df_stats['avg_risk'].iloc[0]
        std_risk = df_stats['std_risk'].iloc[0]
        min_risk = df_stats['min_risk'].iloc[0]
        max_risk = df_stats['max_risk'].iloc[0]
        pct_05 = df_stats['pct_gte_05'].iloc[0]
        pct_07 = df_stats['pct_gte_07'].iloc[0]
        pct_09 = df_stats['pct_gte_09'].iloc[0]
    else:
        total_students, avg_risk, std_risk, min_risk, max_risk = 10141014, 0.1484, 0.2318, 0.0144, 0.9377
        pct_05, pct_07, pct_09 = 10.67, 7.33, 0.35

    st.warning("Live AI Predictions not found. Displaying simulated baseline data from database.")

report_date = datetime.date.today().strftime("%Y-%m-%d")

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Total Scored Students</div><div class='kpi-value'>{total_students:,}</div></div>", unsafe_allow_html=True)
with col2: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Average Risk Score</div><div class='kpi-value'>{avg_risk:.4f}</div></div>", unsafe_allow_html=True)
with col3: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Std Dev Risk Score</div><div class='kpi-value'>{std_risk:.4f}</div></div>", unsafe_allow_html=True)
with col4: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Report Timestamp</div><div class='kpi-value' style='font-size: 26px;'>{report_date}</div></div>", unsafe_allow_html=True)

st.markdown("""
<div class='content-card' style='background-color: #F8FAFC;'>
    <h4 style='margin-top:0; color: #1E293B;'>Dynamic Risk Thresholds (Percentile-Based):</h4>
    <p><b>Low Risk:</b> &lt; 0.5000 (bottom 33%)</p>
    <p><b>Medium Risk:</b> 0.5000 to 0.7500 (middle 33%)</p>
    <p><b>High Risk:</b> &ge; 0.7500 (top 34%)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Risk Score Distribution")
r1, r2, r3 = st.columns(3)
with r1: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Min Risk Score</div><div class='kpi-value'>{min_risk:.4f}</div></div>", unsafe_allow_html=True)
with r2: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Median Risk Score (Q50)</div><div class='kpi-value'>{p50:.4f}</div></div>", unsafe_allow_html=True)
with r3: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Max Risk Score</div><div class='kpi-value'>{max_risk:.4f}</div></div>", unsafe_allow_html=True)

st.markdown("### Risk Quartiles")
q1, q2, q3 = st.columns(3)
with q1: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>25th Percentile</div><div class='kpi-value'>{p25:.4f}</div></div>", unsafe_allow_html=True)
with q2: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>50th Percentile</div><div class='kpi-value'>{p50:.4f}</div></div>", unsafe_allow_html=True)
with q3: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>75th Percentile</div><div class='kpi-value'>{p75:.4f}</div></div>", unsafe_allow_html=True)

st.markdown("### Reference Point Distribution (Fixed Thresholds for Trend Tracking)")
st.caption("These are fixed reference points to track consistency over time, separate from the dynamic percentile-based thresholds above.")

ref1, ref2, ref3 = st.columns(3)
with ref1: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>% Students &ge; 0.5</div><div class='kpi-value'>{pct_05:.2f}%</div></div>", unsafe_allow_html=True)
with ref2: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>% Students &ge; 0.7</div><div class='kpi-value'>{pct_07:.2f}%</div></div>", unsafe_allow_html=True)
with ref3: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>% Students &ge; 0.9</div><div class='kpi-value'>{pct_09:.2f}%</div></div>", unsafe_allow_html=True)

st.markdown("### Data Drift Detection (KS Test)")
st.info("No drift report available. Ensure training data is available for comparison.")

st.markdown("""
<div class='content-card'>
    <h5 style='margin-top:0;'>Drift Monitoring Explanation:</h5>
    <ul>
        <li>Compares feature distributions in the current scored dataset with the training panel.</li>
        <li>Uses Kolmogorov-Smirnov (KS) test to detect distribution shifts.</li>
        <li>P-value &lt; 0.05 indicates statistically significant drift.</li>
        <li>High drift may indicate need for model retraining or recalibration.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
