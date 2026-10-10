import streamlit as st
import pandas as pd
import numpy as np
import math
import sys
import os
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import shap

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

from config.database import fetch_data
from ml_pipeline.preprocess import clean_and_engineer_features, calculate_terminal_transition_risk
from ui_utils import inject_enterprise_css

inject_enterprise_css()

st.markdown("<h1>Student Risk Scores</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>State-wide predictive forecasting and intervention targeting</div>", unsafe_allow_html=True)

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
parquet_path = os.path.join(ARTIFACTS_DIR, 'statewide_scores.parquet')
model_path = os.path.join(ARTIFACTS_DIR, 'best_model.pkl')
json_path = os.path.join(ARTIFACTS_DIR, 'model_comparison.json')
TABLE_NAME = "aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep"

@st.cache_data(ttl=3600)
def load_statewide_scores():
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_total_db_records():
    try:
        query = f"SELECT COUNT(*) as total FROM {TABLE_NAME} WHERE person_id IS NOT NULL AND class_currently_enrolled BETWEEN 1 AND 10"
        df = fetch_data(query)
        if not df.empty:
            return df['total'].iloc[0]
    except Exception:
        pass
    return 10141014 

@st.cache_data(ttl=3600)
def load_school_master():
    try:
        query = f"SELECT DISTINCT school_id, school_category, school_type_coed_boys_girls, school_latitude AS latitude, school_longitude AS longitude FROM {TABLE_NAME} WHERE school_latitude IS NOT NULL"
        return fetch_data(query)
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_ml_model():
    if os.path.exists(model_path) and os.path.exists(json_path):
        model = joblib.load(model_path)
        with open(json_path, 'r') as f:
            meta = json.load(f)
        return model, meta.get("feature_names", []), meta.get("categorical_options", {})
    return None, [], {}

with st.spinner("Loading state-wide risk profiles..."):
    df_all = load_statewide_scores()
    total_db_records = get_total_db_records()
    xgb_model, feature_names, categorical_options = load_ml_model()

if df_all.empty:
    st.warning("Statewide scores not found. Please run 'python ml_pipeline/batch_score.py' in your terminal first.")
    st.stop()

def get_options(column):
    if column in df_all.columns:
        return ["All"] + sorted([str(x) for x in df_all[column].dropna().unique()])
    return ["All"]

if 'page_num' not in st.session_state: st.session_state.page_num = 1
if 'records_per_page' not in st.session_state: st.session_state.records_per_page = 15

def reset_page(): st.session_state.page_num = 1
def clear_filters():
    st.session_state.page_num = 1
    for key in ['risk_filter_ai', 'risk_filter_manual', 'dist_filter', 'class_filter', 'gender_filter', 'income_filter', 'social_filter', 'search_query']:
        if key in st.session_state: del st.session_state[key]

col1, col2, col3, col4 = st.columns(4)
with col1: risk_level_ai = st.selectbox("Risk Level (AI) [High: >=65%]", ["All", "High (>= 65%)", "Medium (35-64%)", "Low (< 35%)"], key='risk_filter_ai', on_change=reset_page)
with col2: risk_level_manual = st.selectbox("Risk Level (Manual) [Critical: >=81]", ["All", "Critical (>= 81)", "High (61-80)", "Medium (31-60)", "Low (<= 30)"], key='risk_filter_manual', on_change=reset_page)
with col3: selected_district = st.selectbox("District", get_options("student_district_name"), key='dist_filter', on_change=reset_page)
with col4: selected_class = st.selectbox("Class", get_options("baseline_class"), key='class_filter', on_change=reset_page)

col5, col6, col7, col8 = st.columns(4)
with col5: selected_gender = st.selectbox("Gender", get_options("student_gender"), key='gender_filter', on_change=reset_page)
with col6: selected_income = st.selectbox("Income Band", get_options("income_band"), key='income_filter', on_change=reset_page)
with col7: selected_social = st.selectbox("Social Category", get_options("social_category"), key='social_filter', on_change=reset_page)
with col8: 
    search_query = st.text_input("Search (ID or School...)", key='search_query', on_change=reset_page)
    if st.button("Clear Filters", type="primary", use_container_width=True):
        clear_filters()
        st.rerun()

st.markdown("---")

filtered_df = df_all.copy()

if risk_level_ai != "All": 
    ai_val = risk_level_ai.split(" ")[0] 
    filtered_df = filtered_df[filtered_df.get('ml_risk_level', filtered_df.get('risk_level')) == ai_val]
if risk_level_manual != "All" and 'stakeholder_risk_level' in filtered_df.columns:
    man_val = risk_level_manual.split(" ")[0]
    filtered_df = filtered_df[filtered_df['stakeholder_risk_level'] == man_val]

if selected_district != "All": filtered_df = filtered_df[filtered_df['student_district_name'].astype(str) == selected_district]
if selected_class != "All": filtered_df = filtered_df[filtered_df['baseline_class'].astype(str) == selected_class]
if selected_gender != "All": filtered_df = filtered_df[filtered_df['student_gender'].astype(str) == selected_gender]
if selected_income != "All": filtered_df = filtered_df[filtered_df['income_band'].astype(str) == selected_income]
if selected_social != "All": filtered_df = filtered_df[filtered_df['social_category'].astype(str) == selected_social]

if search_query:
    sq = search_query.strip().lower()
    filtered_df = filtered_df[
        filtered_df['person_id'].astype(str).str.lower().str.contains(sq) | 
        filtered_df['school_name'].astype(str).str.lower().str.contains(sq)
    ]

total_filtered = len(filtered_df)
discrepancy_rate = (~(filtered_df.get('ml_risk_level') == filtered_df.get('stakeholder_risk_level'))).mean() * 100 if total_filtered > 0 else 0.0

high_risk_count = len(filtered_df[filtered_df.get('ml_risk_level', filtered_df.get('risk_level')) == 'High'])
med_risk_count = len(filtered_df[filtered_df.get('ml_risk_level', filtered_df.get('risk_level')) == 'Medium'])

val_style = "font-size: 22px; white-space: nowrap; letter-spacing: -0.5px;"
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Statewide Records</div><div class='kpi-value' style='{val_style}'>{total_db_records:,}</div></div>", unsafe_allow_html=True)
with kpi2: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Actionable Matches</div><div class='kpi-value' style='{val_style}'>{total_filtered:,}</div></div>", unsafe_allow_html=True)
with kpi3: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>High Risk (AI)</div><div class='kpi-value' style='color: #E11D48; {val_style}'>{high_risk_count:,}</div></div>", unsafe_allow_html=True)
with kpi4: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>Medium Risk (AI)</div><div class='kpi-value' style='color: #D97706; {val_style}'>{med_risk_count:,}</div></div>", unsafe_allow_html=True)
with kpi5: st.markdown(f"<div class='kpi-container'><div class='kpi-title'>AI vs Manual Mismatch</div><div class='kpi-value' style='color: #8B5CF6; {val_style}'>{discrepancy_rate:.1f}%</div></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

total_pages = max(1, math.ceil(total_filtered / st.session_state.records_per_page))
if st.session_state.page_num > total_pages: st.session_state.page_num = total_pages

page_col1, page_col2, page_col3, page_col4, page_col5 = st.columns([1.5, 2.5, 1.5, 3.5, 5])
with page_col1:
    if st.button("< Prev") and st.session_state.page_num > 1:
        st.session_state.page_num -= 1
        st.rerun()
with page_col2: st.markdown(f"<div style='text-align: center; padding-top: 5px; font-weight: bold;'>Page {st.session_state.page_num} of {total_pages}</div>", unsafe_allow_html=True)
with page_col3:
    if st.button("Next >") and st.session_state.page_num < total_pages:
        st.session_state.page_num += 1
        st.rerun()
with page_col4:
    records_per_page = st.selectbox("Records per page", [10, 15, 50, 100, 500], index=1, key='rec_page')
    if records_per_page != st.session_state.records_per_page:
        st.session_state.records_per_page = records_per_page
        st.session_state.page_num = 1
        st.rerun()
with page_col5:
    if total_filtered > 0:
        csv = filtered_df.head(50000).to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Filtered List (CSV)", data=csv, file_name="student_risk_export.csv", mime="text/csv")

display_df = pd.DataFrame()
if total_filtered > 0:
    start_idx = (st.session_state.page_num - 1) * st.session_state.records_per_page
    end_idx = start_idx + st.session_state.records_per_page

    ml_score_col = 'ml_risk_score' if 'ml_risk_score' in filtered_df.columns else 'risk_score'
    ml_level_col = 'ml_risk_level' if 'ml_risk_level' in filtered_df.columns else 'risk_level'

    display_df = filtered_df.sort_values(by=ml_score_col, ascending=False).iloc[start_idx:end_idx].copy()
    
    cols_to_show = ["person_id", "school_name", "baseline_class", "student_district_name", ml_score_col, ml_level_col]
    if 'stakeholder_manual_score' in display_df.columns:
        cols_to_show.extend(["stakeholder_manual_score", "stakeholder_risk_level"])
        
    display_df = display_df[cols_to_show].rename(columns={
        "person_id": "Student ID",
        "school_name": "School Name",
        "baseline_class": "Class",
        "student_district_name": "District",
        ml_score_col: "AI Probability",
        ml_level_col: "AI Tier",
        "stakeholder_manual_score": "Manual Score",
        "stakeholder_risk_level": "Manual Tier"
    })

    def style_risk_score(val):
        if isinstance(val, str):
            if val == 'High' or val == 'Critical': return 'color: #E11D48; font-weight: bold;'
            if val == 'Medium': return 'color: #D97706; font-weight: bold;'
            if val == 'Low': return 'color: #10B981; font-weight: bold;'
        elif isinstance(val, (int, float)):
            if val >= 0.65 or val >= 61: return 'color: #E11D48; font-weight: bold;'
            if val >= 0.35 or val >= 31: return 'color: #D97706; font-weight: bold;'
            return 'color: #10B981; font-weight: bold;'
        return ''

    def highlight_mismatch(row):
        if 'AI Tier' in row and 'Manual Tier' in row:
            if row['AI Tier'] != row['Manual Tier'] and not (row['AI Tier'] == 'High' and row['Manual Tier'] == 'Critical'):
                return ['background-color: rgba(139, 92, 246, 0.1)'] * len(row)
        return [''] * len(row)

    styled_df = display_df.style.apply(highlight_mismatch, axis=1).map(
        style_risk_score, subset=['AI Probability', 'AI Tier']
    )
    
    if 'Manual Score' in display_df.columns:
        styled_df = styled_df.map(style_risk_score, subset=['Manual Score', 'Manual Tier'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.info("No students found matching the selected filters.")

def get_manual_breakdown(row):
    breakdown = []
    
    cls = 0 if pd.isna(row.get('baseline_class')) else row.get('baseline_class', 0)
    p_cls = 100 if cls in [5, 8, 9] else 40 if cls in [6, 7] else 0
    breakdown.append({"Indicator Category": "Academic", "Feature": "Baseline Class", "Value": f"Class {int(cls)}", "Base Penalty": p_cls, "Weight": "8.5%", "Points": p_cls * 0.0850})

    upper_limit = cls + 6
    age = upper_limit if pd.isna(row.get('student_age')) else row.get('student_age', upper_limit)
    gap = age - upper_limit
    p_gap = 100 if gap >= 3 else 60 if gap == 2 else 0
    breakdown.append({"Indicator Category": "Academic", "Feature": "Over-Age Gap", "Value": f"Gap of {gap:.1f} Yrs", "Base Penalty": p_gap, "Weight": "8.0%", "Points": p_gap * 0.0800})

    att = row.get('current_attendance', 100)
    att = 100 if pd.isna(att) else att
    p_att = 100 if att < 50 else 70 if att < 70 else 30 if att < 85 else 0
    breakdown.append({"Indicator Category": "Behavioral", "Feature": "Current Attendance", "Value": f"{att:.1f}%", "Base Penalty": p_att, "Weight": "7.0%", "Points": p_att * 0.0700})

    fam = 100 if pd.isna(row.get('current_fa_marks')) else row.get('current_fa_marks', 100)
    p_fam = 100 if fam < 35 else 70 if fam < 50 else 30 if fam < 75 else 0
    breakdown.append({"Indicator Category": "Academic", "Feature": "Current FA Marks", "Value": f"{fam:.1f}", "Base Penalty": p_fam, "Weight": "6.5%", "Points": p_fam * 0.0650})

    sch = str(row.get('school_category', 'Unknown')).upper()
    p_sch = 100 if 'UNKNOWN' in sch else 5 if sch == 'UPPER PRIMARY ONLY' else 3 if 'PRIMARY' in sch else 0
    breakdown.append({"Indicator Category": "Infrastructure", "Feature": "School Category", "Value": sch.title(), "Base Penalty": p_sch, "Weight": "5.5%", "Points": p_sch * 0.0550})

    smt = 0 if pd.isna(row.get('sa_marks_trend')) else row.get('sa_marks_trend', 0)
    p_smt = 100 if smt <= -10 else 40 if smt <= -9 else 0
    breakdown.append({"Indicator Category": "Academic", "Feature": "SA Marks Trend", "Value": f"{smt:.1f}", "Base Penalty": p_smt, "Weight": "4.5%", "Points": p_smt * 0.0450})

    atr = 0 if pd.isna(row.get('attendance_trend')) else row.get('attendance_trend', 0)
    p_atr = 100 if atr <= -20 else 40 if atr <= -10 else 0
    breakdown.append({"Indicator Category": "Behavioral", "Feature": "Attendance Trend", "Value": f"{atr:.1f}%", "Base Penalty": p_atr, "Weight": "4.5%", "Points": p_atr * 0.0450})

    fmt = 0 if pd.isna(row.get('fa_marks_trend')) else row.get('fa_marks_trend', 0)
    p_fmt = 100 if fmt <= -10 else 40 if fmt <= -9 else 0
    breakdown.append({"Indicator Category": "Academic", "Feature": "FA Marks Trend", "Value": f"{fmt:.1f}", "Base Penalty": p_fmt, "Weight": "4.0%", "Points": p_fmt * 0.0400})

    dst = 0 if pd.isna(row.get('distance_to_school_km')) else row.get('distance_to_school_km', 0)
    p_dst = 100 if dst > 10 else 75 if dst >= 6 else 40 if dst >= 3 else 0
    breakdown.append({"Indicator Category": "Accessibility", "Feature": "Dist to School", "Value": f"{dst:.1f} km", "Base Penalty": p_dst, "Weight": "3.5%", "Points": p_dst * 0.0350})

    tdr = 0 if pd.isna(row.get('teacher_deficit_ratio')) else row.get('teacher_deficit_ratio', 0)
    p_tdr = 100 if tdr >= 0.50 else 50 if tdr >= 0.25 else 0
    breakdown.append({"Indicator Category": "Infrastructure", "Feature": "Teacher Deficit", "Value": f"{tdr*100:.0f}%", "Base Penalty": p_tdr, "Weight": "3.5%", "Points": p_tdr * 0.0350})

    med = str(row.get('medium_of_instruction', 'Unknown')).lower()
    p_med = 100 if med == 'tamil' else 40 if med == 'urdu' else 20 if med in ['english', 'telugu'] else 0
    breakdown.append({"Indicator Category": "Social", "Feature": "Medium of Instruction", "Value": med.title(), "Base Penalty": p_med, "Weight": "3.5%", "Points": p_med * 0.0350})

    sam = 100 if pd.isna(row.get('current_sa_marks')) else row.get('current_sa_marks', 100)
    p_sam = 100 if sam < 35 else 70 if sam < 50 else 30 if sam < 75 else 0
    breakdown.append({"Indicator Category": "Academic", "Feature": "Current SA Marks", "Value": f"{sam:.1f}", "Base Penalty": p_sam, "Weight": "3.0%", "Points": p_sam * 0.0300})

    par = str(row.get('parental_presence', 'Unknown'))
    p_par = 0 if par == 'Both IDs Present' else 50 if par in ['Mother ID Only', 'Father ID Only'] else 100
    breakdown.append({"Indicator Category": "Socio-Economic", "Feature": "Parental Presence", "Value": par, "Base Penalty": p_par, "Weight": "3.0%", "Points": p_par * 0.0300})

    tv = str(row.get('current_thalliki_vandanam_status', 'No')).lower()
    p_tv = 0 if tv == 'yes' else 100
    breakdown.append({"Indicator Category": "Welfare", "Feature": "Thalliki Vandanam Status", "Value": tv.capitalize(), "Base Penalty": p_tv, "Weight": "3.0%", "Points": p_tv * 0.0300})

    dwl = str(row.get('dwelling_type', 'Unknown')).lower()
    p_dwl = 100 if 'flat' in dwl or dwl == 'unknown' else 39 if 'duplex' in dwl else 10 if 'kaccha' in dwl else 0
    breakdown.append({"Indicator Category": "Socio-Economic", "Feature": "Dwelling Type", "Value": dwl.title(), "Base Penalty": p_dwl, "Weight": "2.5%", "Points": p_dwl * 0.0250})

    sib = 0 if pd.isna(row.get('siblings_below_5')) else row.get('siblings_below_5', 0)
    p_sib = 100 if sib >= 2 else 50 if sib == 1 else 0
    breakdown.append({"Indicator Category": "Family", "Feature": "Siblings Below 5 Yrs", "Value": str(int(sib)), "Base Penalty": p_sib, "Weight": "2.0%", "Points": p_sib * 0.0200})

    jvk = 4 if pd.isna(row.get('student_kit_score')) else row.get('student_kit_score', 4)
    p_jvk = 100 if jvk == 0 else 50 if jvk <= 2 else 0
    breakdown.append({"Indicator Category": "Welfare", "Feature": "Student Kit Score", "Value": f"{int(jvk)}/4 Items", "Base Penalty": p_jvk, "Weight": "2.0%", "Points": p_jvk * 0.0200})

    d_next = 0 if pd.isna(row.get('distance_to_next_school_km')) else row.get('distance_to_next_school_km', 0)
    p_dnext = 100 if d_next > 10 else 75 if d_next >= 6 else 40 if d_next >= 3 else 0
    breakdown.append({"Indicator Category": "Geospatial", "Feature": "Dist to Next School", "Value": f"{d_next:.1f} km", "Base Penalty": p_dnext, "Weight": "2.0%", "Points": p_dnext * 0.0200})

    soc = str(row.get('social_category', 'Unknown')).upper()
    p_soc = 0 if soc == 'SC' else 7 if soc == 'BC' else 24 if soc == 'ST' else 45 if soc == 'OC' else 100
    breakdown.append({"Indicator Category": "Social", "Feature": "Social Category", "Value": soc, "Base Penalty": p_soc, "Weight": "1.5%", "Points": p_soc * 0.0150})

    mar = str(row.get('marital_status', 'Unknown')).lower()
    p_mar = 0 if mar in ['unmarried', 'unknown'] else 100
    breakdown.append({"Indicator Category": "Social", "Feature": "Marital Status", "Value": mar.title(), "Base Penalty": p_mar, "Weight": "1.5%", "Points": p_mar * 0.0150})

    hv = 0 if pd.isna(row.get('health_vulnerability_score')) else row.get('health_vulnerability_score', 0)
    p_hv = 100 if hv >= 3 else 70 if hv == 2 else 30 if hv == 1 else 0
    breakdown.append({"Indicator Category": "Health", "Feature": "Health Vulnerability", "Value": f"Score {int(hv)}", "Base Penalty": p_hv, "Weight": "1.5%", "Points": p_hv * 0.0150})

    mem = 4 if pd.isna(row.get('members_count')) else row.get('members_count', 4)
    p_mem = 100 if mem == 1 or mem >= 15 else 50 if 9 <= mem <= 14 else 0
    breakdown.append({"Indicator Category": "Socio-Economic", "Feature": "Members Count", "Value": int(mem), "Base Penalty": p_mem, "Weight": "1.5%", "Points": p_mem * 0.0150})

    inv = 0 if pd.isna(row.get('age_invalid_flag')) else row.get('age_invalid_flag', 0)
    p_inv = 100 if inv == 1 else 0
    breakdown.append({"Indicator Category": "System", "Feature": "Age Invalid", "Value": "Yes" if inv == 1 else "No", "Base Penalty": p_inv, "Weight": "1.5%", "Points": p_inv * 0.0150})

    is_fem = str(row.get('student_gender', '')).lower() == 'female'
    d_girl = 0 if pd.isna(row.get('nearest_distance_to_girl_school')) else row.get('nearest_distance_to_girl_school', 0)
    p_girl = 100 if (is_fem and d_girl > 3) else 0
    breakdown.append({"Indicator Category": "Accessibility", "Feature": "Dist to Girl School (>3km)", "Value": f"{d_girl:.1f} km", "Base Penalty": p_girl, "Weight": "1.0%", "Points": p_girl * 0.0100})

    water = 1 if pd.isna(row.get('has_drinking_water')) else row.get('has_drinking_water', 1)
    breakdown.append({"Indicator Category": "Infrastructure", "Feature": "Drinking Water", "Value": "Yes" if water == 1 else "No", "Base Penalty": (100 if water == 0 else 0), "Weight": "1.0%", "Points": (100 if water == 0 else 0) * 0.0100})

    rat = str(row.get('ration_card_type', 'Unknown')).upper()
    breakdown.append({"Indicator Category": "Socio-Economic", "Feature": "Ration Card", "Value": rat, "Base Penalty": (0 if rat in ['AAY', 'BPL'] else 100), "Weight": "1.0%", "Points": (0 if rat in ['AAY', 'BPL'] else 100) * 0.0100})

    ru = str(row.get('rural_urban_flag', 'Unknown')).upper()
    breakdown.append({"Indicator Category": "Accessibility", "Feature": "Rural/Urban", "Value": ru, "Base Penalty": (0 if ru == 'R' else 100), "Weight": "1.0%", "Points": (0 if ru == 'R' else 100) * 0.0100})

    trb = 0 if pd.isna(row.get('is_tribal')) else row.get('is_tribal', 0)
    breakdown.append({"Indicator Category": "Geospatial", "Feature": "Tribal Mandal", "Value": "Yes" if trb == 1 else "No", "Base Penalty": (100 if trb == 1 else 0), "Weight": "1.0%", "Points": (100 if trb == 1 else 0) * 0.0100})

    inc = str(row.get('income_band', 'Unknown')).replace(' ', '')
    p_inc = 0 if inc == '>5L' else 10 if inc == '3L-5L' else 25 if inc == '0-1L' else 33 if inc == '1L-3L' else 100
    breakdown.append({"Indicator Category": "Socio-Economic", "Feature": "Income Band", "Value": inc, "Base Penalty": p_inc, "Weight": "1.0%", "Points": p_inc * 0.0100})

    g_toilet = 1 if pd.isna(row.get('has_girls_toilet')) else row.get('has_girls_toilet', 1)
    breakdown.append({"Indicator Category": "Infrastructure", "Feature": "Girls Toilet", "Value": "Yes" if g_toilet == 1 else "No", "Base Penalty": (100 if (is_fem and g_toilet == 0) else 0), "Weight": "1.0%", "Points": (100 if (is_fem and g_toilet == 0) else 0) * 0.0100})

    gen = str(row.get('student_gender', 'Unknown')).lower()
    p_gen = 0 if gen == 'female' else 4 if gen == 'male' else 31 if gen == 'transgender' else 100
    breakdown.append({"Indicator Category": "Social", "Feature": "Gender", "Value": gen.title(), "Base Penalty": p_gen, "Weight": "1.0%", "Points": p_gen * 0.0100})

    rej = 0 if pd.isna(row.get('is_recent_rejoiner')) else row.get('is_recent_rejoiner', 0)
    breakdown.append({"Indicator Category": "Behavioral", "Feature": "Recent Rejoiner", "Value": "Yes" if rej == 1 else "No", "Base Penalty": (100 if rej == 1 else 0), "Weight": "1.0%", "Points": (100 if rej == 1 else 0) * 0.0100})

    hmy = 0 if pd.isna(row.get('historical_missing_years')) else row.get('historical_missing_years', 0)
    breakdown.append({"Indicator Category": "Profile", "Feature": "Missing Years", "Value": int(hmy), "Base Penalty": (100 if hmy >= 2 else 60 if hmy == 1 else 0), "Weight": "1.0%", "Points": (100 if hmy >= 2 else 60 if hmy == 1 else 0) * 0.0100})

    t_risk = 0 if pd.isna(row.get('is_terminal_transition_risk')) else row.get('is_terminal_transition_risk', 0)
    breakdown.append({"Indicator Category": "Geospatial", "Feature": "Terminal Transition Risk", "Value": "Yes" if t_risk == 1 else "No", "Base Penalty": (100 if t_risk == 1 else 0), "Weight": "1.0%", "Points": (100 if t_risk == 1 else 0) * 0.0100})

    req_acc = 0 if pd.isna(row.get('requires_accessible_school')) else row.get('requires_accessible_school', 0)
    p_req = 100 if req_acc == 1 else 0
    breakdown.append({"Indicator Category": "Special Needs", "Feature": "Requires Accessible School", "Value": "Yes" if req_acc == 1 else "No", "Base Penalty": p_req, "Weight": "5.0%", "Points": p_req * 0.0500})

    return pd.DataFrame(breakdown)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("### Dual-Score Student Explanation")
st.markdown("Select a student ID below to view a deep-dive explanation of both their AI Risk Score and their Stakeholder Manual Score.")

if not display_df.empty and xgb_model:
    visible_ids = display_df['Student ID'].astype(str).tolist()
    selected_id = st.selectbox("Choose a student...", [""] + visible_ids)

    if selected_id:
        with st.spinner("Calculating live SHAP and Matrix contributions..."):
            query = f"SELECT * FROM {TABLE_NAME} WHERE CAST(person_id AS STRING) = '{selected_id}' AND class_currently_enrolled BETWEEN 1 AND 10"
            raw_df = fetch_data(query)

            if not raw_df.empty:
                raw_df = raw_df.rename(columns={
                    'class_currently_enrolled': 'baseline_class',
                    'thalliki_vandanam_26': 'current_thalliki_vandanam_status',
                    'academic_26': 'baseline_status',
                    'academic_25': 'historical_status_1',
                    'academic_24': 'historical_status_2',
                    'academic_23': 'historical_status_3',
                    'attendance_percentage_2025_2026': 'current_attendance',
                    'attendance_percentage_2024_2025': 'previous_attendance',
                    'fa_2025_2026': 'current_fa_marks',
                    'fa_2024_2025': 'previous_fa_marks',
                    'sa_2025_2026': 'current_sa_marks',
                    'sa_2024_2025': 'previous_sa_marks'
                })
                raw_df['target_status'] = np.nan 
                
                proc_df = clean_and_engineer_features(raw_df.copy())
                
                school_df = load_school_master()
                if not school_df.empty:
                    proc_df = calculate_terminal_transition_risk(proc_df, school_df)
                else:
                    proc_df['distance_to_next_school_km'] = np.nan
                    proc_df['nearest_distance_to_girl_school'] = np.nan
                    proc_df['is_terminal_transition_risk'] = 0
                
                tab1, tab2 = st.tabs([" AI Model (SHAP Explainer)", " Stakeholder Manual Matrix (Rule Explainer)"])
                
                with tab1:
                    for c in feature_names:
                        if c not in proc_df.columns: 
                            proc_df[c] = np.nan

                    X_live = proc_df[feature_names].copy()
                    
                    categorical_cols = [
                        'student_gender', 'social_category', 'income_band', 'dwelling_type', 
                        'ration_card_type', 'parental_presence', 'school_category', 
                        'current_thalliki_vandanam_status', 'rural_urban_flag', 'marital_status', 'medium_of_instruction'
                    ]
                    
                    for c in X_live.columns:
                        if c in categorical_cols:
                            X_live[c] = X_live[c].astype('category')
                        else:
                            X_live[c] = pd.to_numeric(X_live[c], errors='coerce')

                    if hasattr(xgb_model, 'named_steps'):
                        processed_input = xgb_model.named_steps['preprocessor'].transform(X_live)
                        try:
                            explainer = joblib.load(os.path.join(ARTIFACTS_DIR, 'shap_explainer.pkl'))
                            shap_values = explainer.shap_values(processed_input)
                            base_value = explainer.expected_value
                            if isinstance(base_value, (list, np.ndarray)): base_value = base_value[-1]
                        except:
                            st.warning("SHAP explanation not fully compatible with pipeline models in UI context yet.")
                            shap_values = np.zeros(X_live.shape)
                            base_value = 0.5
                    else:
                        explainer = shap.TreeExplainer(xgb_model)
                        try:
                            shap_values = explainer.shap_values(X_live)
                        except Exception as e:
                            # --- FALLBACK: Translating categorical strings to numeric encodings for Scikit-Learn ---
                            if "could not convert string to float" in str(e):
                                X_shap = X_live.copy()
                                for c in categorical_cols:
                                    if c in X_shap.columns and c in categorical_options:
                                        X_shap[c] = pd.Categorical(X_shap[c], categories=categorical_options[c])
                                        X_shap[c] = X_shap[c].cat.codes
                                shap_values = explainer.shap_values(X_shap)
                            else:
                                raise e
                                
                        base_value = explainer.expected_value
                        if isinstance(base_value, (list, np.ndarray)): base_value = base_value[-1]

                    if hasattr(xgb_model, "predict_proba"):
                        pred_prob = xgb_model.predict_proba(X_live)[0][1]
                    else:
                        pred_prob = xgb_model.predict(X_live)[0]

                    shap_val_array = shap_values[0] if isinstance(shap_values, list) else shap_values[0] if hasattr(shap_values, 'shape') else shap_values
                    if len(shap_val_array.shape) > 1: shap_val_array = shap_val_array[:, 1]

                    shap_df = pd.DataFrame({
                        'feature': feature_names,
                        'value': X_live.iloc[0].values,
                        'shap_value': shap_val_array
                    })
                    shap_df['label'] = shap_df['feature'].str.replace('_', ' ').str.title()
                    shap_df['impact'] = np.where(shap_df['shap_value'] > 0, 'Increase Risk', 'Decrease Risk')
                    shap_df['abs_shap'] = shap_df['shap_value'].abs()
                    
                    total_abs_shap = shap_df['abs_shap'].sum()
                    shap_df['impact_pct'] = np.where(total_abs_shap > 0, (shap_df['abs_shap'] / total_abs_shap) * 100, 0)

                    shap_df = shap_df.sort_values(by='abs_shap', ascending=False).head(10)

                    inc_factors = shap_df[shap_df['shap_value'] > 0]
                    dec_factors = shap_df[shap_df['shap_value'] < 0]

                    inc_text = ", ".join([f"{r['label']} ({r['impact_pct']:.1f}%)" for _, r in inc_factors.iterrows()])
                    dec_text = ", ".join([f"{r['label']} ({r['impact_pct']:.1f}%)" for _, r in dec_factors.iterrows()])

                    st.write(f"**Student Record ID:** {selected_id}")
                    st.write(f"**Predicted Dropout Risk (AI):** {pred_prob*100:.1f}%")
                    st.write(f"**Base model output (expected value):** {base_value:.4f}")

                    st.markdown("**Explanation (plain English):**")
                    full_text = f"This student is at higher risk primarily because of {inc_text}. " if inc_text else "This student has a lower baseline risk. "
                    if dec_text:
                        full_text += f"Factors acting as safety nets reducing risk include {dec_text}."
                    st.write(full_text)

                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        color_map = {'Increase Risk': '#E11D48', 'Decrease Risk': '#10B981'}
                        
                        fig_pct = go.Figure(go.Bar(
                            x=shap_df['label'],
                            y=shap_df['impact_pct'],
                            marker_color=shap_df['impact'].map(color_map),
                            text=shap_df['impact_pct'].apply(lambda x: f"{x:.1f}%"),
                            textposition='outside'
                        ))
                        fig_pct.update_layout(
                            title="Actionable Risk Drivers (% Impact)", 
                            template="plotly_white",
                            margin=dict(l=0, r=0, b=0, t=40),
                            yaxis_title="Proportional Impact (%)"
                        )
                        fig_pct.update_yaxes(range=[0, shap_df['impact_pct'].max() * 1.2])
                        fig_pct.update_xaxes(tickangle=-90)
                        st.plotly_chart(fig_pct, use_container_width=True)

                    with chart_col2:
                        fig_raw = go.Figure(go.Bar(
                            x=shap_df['label'],
                            y=shap_df['shap_value'],
                            marker_color='#026BC1'
                        ))
                        fig_raw.update_layout(
                            title="Data Science View (Raw Log-Odds)", 
                            template="plotly_white",
                            margin=dict(l=0, r=0, b=0, t=40),
                            yaxis_title="SHAP Value"
                        )
                        fig_raw.update_xaxes(tickangle=-90)
                        st.plotly_chart(fig_raw, use_container_width=True)

                    display_shap = shap_df[['label', 'feature', 'value', 'impact', 'impact_pct', 'shap_value']].copy()
                    display_shap['impact_pct'] = display_shap['impact_pct'].apply(lambda x: f"{x:.2f}%")
                    display_shap['shap_value'] = display_shap['shap_value'].round(4)
                    
                    display_shap = display_shap.rename(columns={
                        'label': 'UI Label', 
                        'feature': 'Database Column', 
                        'value': 'Student Value', 
                        'impact': 'Direction',
                        'impact_pct': 'Relative Impact (%)',
                        'shap_value': 'Raw Log-Odds'
                    })
                    
                    st.dataframe(display_shap, use_container_width=True, hide_index=True)

                with tab2:
                    with st.expander("?? Understanding Engineered Composite Scores"):
                        st.markdown("""
                        **1. Health Vulnerability Score (Max 4 Points)**
                        Calculates a student's biological risk profile based on existing health conditions:
                        * **Hemoglobin (Hb) Status:** Adds 1 point if marked as 'Severe' or 'Moderate'.
                        * **Abnormal Condition:** Adds 1 point if any abnormal health condition is flagged.
                        * **Anemic Flag:** Adds 1 point if the student is diagnosed as anemic.
                        * **4D Condition Flag:** Adds 1 point if the student falls under 'Childhood Diseases', 'Defects At Birth', 'Developmental Delays', or 'Deficiencies'.

                        **2. Student Kit Score (Max 4 Points)**
                        Tracks welfare distribution status. A score of 4 means fully equipped. Missing items rapidly increase risk penalty.
                        * **Uniform Received:** 1 Point
                        * **Textbooks Received:** 1 Point
                        * **Bags Received:** 1 Point
                        * **Shoes Received:** 1 Point

                        **3. Requires Accessible School Flag (Binary: Yes or No)**
                        A unified structural identifier. It flags as `Yes` if the student has *any* of the following conditions, streamlining model feature importance:
                        * Blindness
                        * Orthopaedically Handicapped
                        * Mentally Retarded
                        * Deaf and Dumb
                        * Juvenile Needs
                        """)

                    row_dict = proc_df.iloc[0].to_dict()
                    breakdown_df = get_manual_breakdown(row_dict)
                    
                    total_manual_score = breakdown_df['Points'].sum()
                    top_factors = breakdown_df[breakdown_df['Points'] > 0].sort_values(by='Points', ascending=False).head(5)
                    factors_text = ", ".join([f"**{r['Feature']}** (+{r['Points']:.1f} pts)" for _, r in top_factors.iterrows()])
                    
                    st.write(f"**Student Record ID:** {selected_id}")
                    st.write(f"**Total Stakeholder Manual Score:** {total_manual_score:.1f} / 100")
                    
                    st.markdown("**Score Construction (plain English):**")
                    if factors_text:
                        st.write(f"The largest factors driving this manual risk score are {factors_text}.")
                    else:
                        st.write("This student has triggered no significant manual risk rules.")
                    
                    fig_manual = go.Figure(go.Bar(
                        x=breakdown_df[breakdown_df['Points'] > 0].sort_values(by='Points', ascending=False)['Feature'],
                        y=breakdown_df[breakdown_df['Points'] > 0].sort_values(by='Points', ascending=False)['Points'],
                        marker_color='#8B5CF6'
                    ))
                    fig_manual.update_layout(
                        title="Stakeholder Matrix Penalty Points (By Feature)", 
                        template="plotly_white",
                        margin=dict(l=0, r=0, b=0, t=40),
                        yaxis_title="Points Contributed"
                    )
                    fig_manual.update_xaxes(tickangle=-90)
                    st.plotly_chart(fig_manual, use_container_width=True)
                    
                    display_manual = breakdown_df.sort_values(by='Points', ascending=False).copy()
                    display_manual['Points'] = display_manual['Points'].apply(lambda x: f"+{x:.1f}")
                    st.dataframe(display_manual[['Indicator Category', 'Feature', 'Value', 'Base Penalty', 'Weight', 'Points']], use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.markdown("### Risk Points by Indicator Category")
                    
                    category_df = breakdown_df.groupby('Indicator Category')['Points'].sum().reset_index()
                    category_df = category_df[category_df['Points'] > 0].sort_values(by='Points', ascending=False)
                    
                    if not category_df.empty:
                        fig_pie = px.pie(category_df, values='Points', names='Indicator Category', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        st.markdown("#### Category Feature Breakdown")
                        selected_cat = st.selectbox("Select an Indicator Category to view its specific feature contributions:", options=category_df['Indicator Category'].unique())
                        if selected_cat:
                            cat_features = breakdown_df[breakdown_df['Indicator Category'] == selected_cat].sort_values(by='Points', ascending=False)
                            fig_cat_bar = px.bar(cat_features, x='Feature', y='Points', color_discrete_sequence=['#F59E0B'])
                            fig_cat_bar.update_layout(title=f"{selected_cat} Distributions", template="plotly_white", margin=dict(l=0, r=0, b=0, t=40), yaxis_title="Points Contributed", xaxis_title="")
                            st.plotly_chart(fig_cat_bar, use_container_width=True)
                            
                            display_cat = cat_features.copy()
                            display_cat['Points'] = display_cat['Points'].apply(lambda x: f"+{x:.1f}")
                            st.dataframe(display_cat[['Feature', 'Value', 'Base Penalty', 'Weight', 'Points']], use_container_width=True, hide_index=True)
                    else:
                        st.info("This student has 0 penalty points across all indicator categories.")
            else:
                st.warning("Could not fetch raw data for this student.")