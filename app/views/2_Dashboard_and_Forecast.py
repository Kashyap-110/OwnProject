import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

from config.database import fetch_data
from ui_utils import inject_enterprise_css

inject_enterprise_css()

st.markdown("<h1>Dashboard & Forecast</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>District risk map, demographics, and multi-year dropout forecasting</div>", unsafe_allow_html=True)

st.markdown("## Multi-Year Dropout Forecasting")

@st.cache_data(ttl=3600)
def get_historical_dropout_rate():
    query = """
    WITH yearly_stats AS (
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN (LOWER(CAST(academic_23 AS STRING)) IN ('no', 'na', 'none', 'null', '') OR academic_23 IS NULL) AND COALESCE(ssdc_active, 0) != 1 AND UPPER(COALESCE(course_group_desc, '')) NOT LIKE '%ITI%' AND UPPER(COALESCE(ub_name, '')) NOT LIKE '%ITI%' THEN 1 ELSE 0 END) as drops
        FROM aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep
        WHERE LOWER(CAST(academic_22 AS STRING)) = 'yes' AND class_currently_enrolled BETWEEN 1 AND 10
        
        UNION ALL
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN (LOWER(CAST(academic_24 AS STRING)) IN ('no', 'na', 'none', 'null', '') OR academic_24 IS NULL) AND COALESCE(ssdc_active, 0) != 1 AND UPPER(COALESCE(course_group_desc, '')) NOT LIKE '%ITI%' AND UPPER(COALESCE(ub_name, '')) NOT LIKE '%ITI%' THEN 1 ELSE 0 END) as drops
        FROM aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep
        WHERE LOWER(CAST(academic_23 AS STRING)) = 'yes' AND class_currently_enrolled BETWEEN 1 AND 10
        
        UNION ALL
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN (LOWER(CAST(academic_25 AS STRING)) IN ('no', 'na', 'none', 'null', '') OR academic_25 IS NULL) AND COALESCE(ssdc_active, 0) != 1 AND UPPER(COALESCE(course_group_desc, '')) NOT LIKE '%ITI%' AND UPPER(COALESCE(ub_name, '')) NOT LIKE '%ITI%' THEN 1 ELSE 0 END) as drops
        FROM aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep
        WHERE LOWER(CAST(academic_24 AS STRING)) = 'yes' AND class_currently_enrolled BETWEEN 1 AND 10
        
        UNION ALL
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN (LOWER(CAST(academic_26 AS STRING)) IN ('no', 'na', 'none', 'null', '') OR academic_26 IS NULL) AND COALESCE(ssdc_active, 0) != 1 AND UPPER(COALESCE(course_group_desc, '')) NOT LIKE '%ITI%' AND UPPER(COALESCE(ub_name, '')) NOT LIKE '%ITI%' THEN 1 ELSE 0 END) as drops
        FROM aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep
        WHERE LOWER(CAST(academic_25 AS STRING)) = 'yes' AND class_currently_enrolled BETWEEN 1 AND 10
    )
    SELECT SUM(drops) * 1.0 / NULLIF(SUM(total), 0) AS historical_rate FROM yearly_stats
    """
    try:
        df = fetch_data(query)
        rate = float(df['historical_rate'].iloc[0])
        return max(0.001, min(rate, 0.05))
    except:
        return 0.0055

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
parquet_path = os.path.join(ARTIFACTS_DIR, 'statewide_scores.parquet')

@st.cache_data(ttl=3600)
def load_statewide_scores():
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()

with st.spinner("Loading baseline forecast data and calibrating AI thresholds..."):
    df_live = load_statewide_scores()

if df_live.empty:
    st.warning("Statewide scores not found. Please run 'batch_score.py' first.")
    st.stop()

ai_score_col = 'ml_risk_score' if 'ml_risk_score' in df_live.columns else 'risk_score'

if ai_score_col not in df_live.columns:
    st.error(f"No '{ai_score_col}' column found in the dataset.")
    st.stop()

historical_rate = get_historical_dropout_rate()
target_quantile = 1.0 - historical_rate

dynamic_threshold = df_live[ai_score_col].quantile(target_quantile)
df_live['is_forecasted_dropout'] = (df_live[ai_score_col] >= dynamic_threshold).astype(int)

df_live['student_district_name'] = df_live['student_district_name'].fillna('Unknown')
df_live['baseline_class'] = df_live['baseline_class'].fillna('Unknown').astype(str)
df_live['student_gender'] = df_live['student_gender'].fillna('Unknown')

st.info(f"**AI Calibration Active:** Anchored to a multi-year baseline dropout rate of **{historical_rate*100:.2f}%**. Only students with a predictive risk score above **{dynamic_threshold:.4f}** are forecasted as dropouts.")

forecast_view = st.radio("Forecast View", ["State-wide", "District-wise", "Class-wise", "Gender-wise"], horizontal=True)

forecast_years = ["2026-27", "2027-28", "2028-29", "2029-30"]
selected_year = st.selectbox("Select Forecast Year", forecast_years)

year_index = forecast_years.index(selected_year)

def project_future_totals(df, years_out):
    current_pop = len(df)
    base_dropouts = df['is_forecasted_dropout'].sum()
    group_rate = base_dropouts / current_pop if current_pop > 0 else 0
    current_dropouts = base_dropouts
    
    for _ in range(years_out):
        current_pop = current_pop - current_dropouts
        if current_pop <= 0:
            return 0, 0
        group_rate = group_rate * 0.95
        current_dropouts = current_pop * group_rate
        
    return current_pop, current_dropouts

if forecast_view == "State-wide":
    projections = []
    for i, yr in enumerate(forecast_years):
        proj_pop, proj_dropouts = project_future_totals(df_live, i)
        rate = (proj_dropouts / proj_pop * 100) if proj_pop > 0 else 0
        projections.append({
            "academic_year": yr,
            "projected_students": proj_pop,
            "predicted_dropouts": proj_dropouts,
            "dropout_rate_pct": rate
        })
    
    proj_df = pd.DataFrame(projections)
    selected_stats = proj_df[proj_df['academic_year'] == selected_year].iloc[0]
    
    val_style = "font-size: 32px; white-space: nowrap; letter-spacing: -1px; color: #1F2937;"
    k1, k2, k3 = st.columns(3)
    with k1: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>{selected_year} - Projected Students</div><div class='kpi-value' style='{val_style}'>{int(selected_stats['projected_students']):,}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Predicted Dropouts</div><div class='kpi-value' style='{val_style}'>{int(selected_stats['predicted_dropouts']):,}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Expected Dropout Rate</div><div class='kpi-value' style='{val_style}'>{selected_stats['dropout_rate_pct']:.2f}%</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    display_df = proj_df.copy()
    display_df['projected_students'] = display_df['projected_students'].round(0).astype(int)
    display_df['predicted_dropouts'] = display_df['predicted_dropouts'].round(0).astype(int)
    display_df['dropout_rate_pct'] = display_df['dropout_rate_pct'].round(4)
    st.dataframe(display_df, use_container_width=True, hide_index=False)
    
    fig = px.line(proj_df, x="academic_year", y="predicted_dropouts", markers=True, color_discrete_sequence=["#3B82F6"])
    fig.update_layout(plot_bgcolor="white", xaxis_title="", yaxis_title="")
    fig.update_yaxes(range=[0, proj_df['predicted_dropouts'].max() * 1.2])
    st.plotly_chart(fig, use_container_width=True)

else:
    group_col_map = {
        "District-wise": "student_district_name",
        "Class-wise": "baseline_class",
        "Gender-wise": "student_gender"
    }
    target_col = group_col_map[forecast_view]
    group_projections = []
    grouped = df_live.groupby(target_col)
    
    for name, group in grouped:
        proj_pop, proj_dropouts = project_future_totals(group, year_index)
        if proj_pop > 0:
            group_projections.append({
                target_col: name,
                "student_count": proj_pop,
                "expected_dropouts": proj_dropouts
            })
            
    group_df = pd.DataFrame(group_projections)
    
    if forecast_view == "District-wise":
        group_df = group_df.rename(columns={"student_district_name": "district_code"})
    elif forecast_view == "Class-wise":
        group_df = group_df.rename(columns={"baseline_class": "class_currently_enrolled"})
        group_df['class_currently_enrolled'] = pd.to_numeric(group_df['class_currently_enrolled'], errors='coerce')
        group_df = group_df.sort_values(by="class_currently_enrolled", ascending=False)
    elif forecast_view == "Gender-wise":
        group_df = group_df.rename(columns={"student_gender": "gender"})
        
    if forecast_view != "Class-wise":
         group_df = group_df.sort_values(by="expected_dropouts", ascending=False)
         
    group_df['student_count'] = group_df['student_count'].round(0).astype(int)
    group_df['expected_dropouts'] = group_df['expected_dropouts'].round(0).astype(int)
    
    st.dataframe(group_df, use_container_width=True, hide_index=False)