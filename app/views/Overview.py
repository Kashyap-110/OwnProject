import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

from config.database import fetch_data
from ui_utils import inject_enterprise_css

inject_enterprise_css()

st.markdown("<h1>Overview Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>District, class, and gender-wise dropout statistics</div>", unsafe_allow_html=True)

TABLE_NAME = "aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep"

def get_year_conditions(year):
    # Map UI year to the exact columns based on your Excel logic
    if year == '2026-27':
        curr_col, prev_col, older_cols = 'academic_26', 'academic_25', ['academic_24', 'academic_23', 'academic_22']
    elif year == '2025-26':
        curr_col, prev_col, older_cols = 'academic_25', 'academic_24', ['academic_23', 'academic_22']
    elif year == '2024-25':
        curr_col, prev_col, older_cols = 'academic_24', 'academic_23', ['academic_22']
    elif year == '2023-24':
        curr_col, prev_col, older_cols = 'academic_23', 'academic_22', []
    elif year == '2022-23':
        curr_col, prev_col, older_cols = 'academic_22', None, []
    else:
        curr_col, prev_col, older_cols = 'academic_26', 'academic_25', ['academic_24', 'academic_23', 'academic_22']

    # SQL Helper Functions
    def is_yes(col): return f"LOWER({col}) IN ('yes', 'Yes', 'YES')"
    def is_no(col): return f"({col} IS NULL OR LOWER({col}) IN ('no', 'No', 'NO', '', 'na', 'none', 'null', 'na'))"

    # 1. Total Enrolled (Base Active condition for the selected year)
    active_expr = f"CASE WHEN {is_yes(curr_col)} THEN 1 ELSE 0 END"

    # 2. Retained
    if prev_col:
        retained_expr = f"CASE WHEN {is_yes(prev_col)} AND {is_yes(curr_col)} THEN 1 ELSE 0 END"
    else:
        retained_expr = active_expr

    # 3. Dropout
    if prev_col:
        dropout_expr = f"""
            CASE 
                WHEN COALESCE(ssdc_active, 0) = 1 THEN 0
                WHEN UPPER(COALESCE(course_group_desc, '')) LIKE '%ITI%' THEN 0
                WHEN UPPER(COALESCE(ub_name, '')) LIKE '%ITI%' THEN 0
                WHEN {is_yes(prev_col)} AND {is_no(curr_col)} THEN 1 
                ELSE 0 
            END
        """
        # Optimize WHERE clause: Only load rows that are present this year OR were present last year
        where_filter = f"({is_yes(curr_col)} OR {is_yes(prev_col)})"
    else:
        dropout_expr = "0"
        where_filter = f"({is_yes(curr_col)})"

    # 4. New
    if prev_col:
        all_prev_no = " AND ".join([is_no(c) for c in [prev_col] + older_cols])
        new_expr = f"CASE WHEN {all_prev_no} AND {is_yes(curr_col)} THEN 1 ELSE 0 END"
    else:
        new_expr = "0"

    # 5. Rejoiner
    if older_cols:
        any_older_yes = " OR ".join([is_yes(c) for c in older_cols])
        rejoiner_expr = f"CASE WHEN {is_no(prev_col)} AND ({any_older_yes}) AND {is_yes(curr_col)} THEN 1 ELSE 0 END"
    else:
        rejoiner_expr = "0"

    return where_filter, active_expr, retained_expr, dropout_expr, new_expr, rejoiner_expr

@st.cache_data(ttl=3600)
def get_kpis(year):
    where_filter, active_expr, retained_expr, dropout_expr, new_expr, rejoiner_expr = get_year_conditions(year)
    query = f"""
    SELECT 
        SUM({active_expr}) as total_active,
        SUM({retained_expr}) as total_retained,
        SUM({dropout_expr}) as total_dropouts,
        SUM({rejoiner_expr}) as total_rejoiners,
        SUM({new_expr}) as total_new,
        COUNT(DISTINCT CASE WHEN {active_expr} = 1 THEN student_district_id END) as total_districts
    FROM {TABLE_NAME}
    WHERE person_id IS NOT NULL 
      AND class_currently_enrolled BETWEEN 1 AND 10
      AND {where_filter}
    """
    return fetch_data(query)

@st.cache_data(ttl=3600)
def get_district_dropouts(year):
    where_filter, _, _, dropout_expr, _, _ = get_year_conditions(year)
    query = f"""
    SELECT student_district_name AS district, 
           SUM({dropout_expr}) AS dropout_count
    FROM {TABLE_NAME}
    WHERE person_id IS NOT NULL 
      AND class_currently_enrolled BETWEEN 1 AND 10
      AND {where_filter}
    GROUP BY student_district_name
    HAVING SUM({dropout_expr}) > 0
    ORDER BY district ASC
    """
    return fetch_data(query)

@st.cache_data(ttl=3600)
def get_class_dropouts(year):
    where_filter, _, _, dropout_expr, _, _ = get_year_conditions(year)
    query = f"""
    SELECT class_currently_enrolled AS baseline_class, 
           SUM({dropout_expr}) AS dropout_count
    FROM {TABLE_NAME}
    WHERE person_id IS NOT NULL 
      AND class_currently_enrolled BETWEEN 1 AND 10
      AND {where_filter}
    GROUP BY class_currently_enrolled
    HAVING SUM({dropout_expr}) > 0
    ORDER BY CAST(class_currently_enrolled AS INT) ASC
    """
    return fetch_data(query)

@st.cache_data(ttl=3600)
def get_gender_dropouts(year):
    where_filter, _, _, dropout_expr, _, _ = get_year_conditions(year)
    query = f"""
    SELECT COALESCE(student_gender, 'Unknown') AS gender, 
           SUM({dropout_expr}) AS dropout_count
    FROM {TABLE_NAME}
    WHERE person_id IS NOT NULL 
      AND class_currently_enrolled BETWEEN 1 AND 10
      AND {where_filter}
    GROUP BY COALESCE(student_gender, 'Unknown')
    HAVING SUM({dropout_expr}) > 0
    ORDER BY dropout_count DESC
    """
    return fetch_data(query)

available_years = ["2026-27", "2025-26", "2024-25", "2023-24", "2022-23"]
academic_year = st.selectbox("Select Academic Year", available_years, index=1)

with st.spinner("Calculating live metrics..."):
    kpi_df = get_kpis(academic_year)

total_students_raw = int(kpi_df['total_active'].iloc[0]) if not kpi_df.empty else 0
retained_raw = int(kpi_df['total_retained'].iloc[0]) if not kpi_df.empty else 0
dropout_raw = int(kpi_df['total_dropouts'].iloc[0]) if not kpi_df.empty else 0
rejoiner_raw = int(kpi_df['total_rejoiners'].iloc[0]) if not kpi_df.empty else 0
new_raw = int(kpi_df['total_new'].iloc[0]) if not kpi_df.empty else 0
districts_raw = int(kpi_df['total_districts'].iloc[0]) if not kpi_df.empty else 0

total_students = f"{total_students_raw:,}"
retained_students = f"{retained_raw:,}"
dropout_students = f"{dropout_raw:,}"
rejoiner_students = f"{rejoiner_raw:,}"
new_students = f"{new_raw:,}"
total_districts = f"{districts_raw:,}"

html_cards = f"""
<style>
.kpi-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 15px; margin-bottom: 25px; }}
.kpi-card {{ background: #ffffff; border: 1px solid #e0e6ed; border-radius: 8px; padding: 15px 5px; text-align: center; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.05); display: flex; flex-direction: column; justify-content: center; height: 115px; }}
.kpi-title {{ color: #4a5568; font-size: 13px; font-weight: 600; margin-bottom: 8px; line-height: 1.3; }}
.kpi-value {{ font-size: 21px; font-weight: 700; letter-spacing: -0.5px; white-space: nowrap; }}
.val-total {{ color: #111827; }}
.val-retained {{ color: #2563eb; }}
.val-rejoiner {{ color: #d97706; }}
.val-new {{ color: #059669; }}
.val-dropout {{ color: #dc2626; }}
.val-districts {{ color: #111827; }}
</style>

<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">Total<br>Students</div><div class="kpi-value val-total">{total_students}</div></div>
    <div class="kpi-card"><div class="kpi-title">Retained<br>Students</div><div class="kpi-value val-retained">{retained_students}</div></div>
    <div class="kpi-card"><div class="kpi-title">Rejoiner<br>Students</div><div class="kpi-value val-rejoiner">{rejoiner_students}</div></div>
    <div class="kpi-card"><div class="kpi-title">New<br>Students</div><div class="kpi-value val-new">{new_students}</div></div>
    <div class="kpi-card"><div class="kpi-title">Dropout<br>Students</div><div class="kpi-value val-dropout">{dropout_students}</div></div>
    <div class="kpi-card"><div class="kpi-title">Total<br>Districts</div><div class="kpi-value val-districts">{total_districts}</div></div>
</div>
"""
st.markdown(html_cards, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### District-wise Dropout Counts")
district_data = get_district_dropouts(academic_year)
st.dataframe(district_data, use_container_width=True, hide_index=True)

if not district_data.empty:
    fig_district = px.bar(district_data, x="district", y="dropout_count", color_discrete_sequence=["#026BC1"])
    fig_district.update_layout(xaxis_title="", yaxis_title="", plot_bgcolor="white", height=400, margin=dict(t=10, l=10, r=10, b=50), xaxis_tickangle=-90)
    st.plotly_chart(fig_district, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
col_bottom1, col_bottom2 = st.columns(2)

with col_bottom1:
    st.markdown("### Class-wise Dropouts")
    class_data = get_class_dropouts(academic_year)
    st.dataframe(class_data, use_container_width=True, hide_index=True)
    if not class_data.empty:
        class_data['baseline_class'] = class_data['baseline_class'].astype(str)
        fig_class = px.bar(class_data, x="baseline_class", y="dropout_count", color_discrete_sequence=["#026BC1"])
        fig_class.update_xaxes(categoryorder='array', categoryarray=class_data['baseline_class'])
        fig_class.update_layout(plot_bgcolor="white", height=350, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig_class, use_container_width=True)

with col_bottom2:
    st.markdown("### Gender-wise Dropouts")
    gender_data = get_gender_dropouts(academic_year)
    st.dataframe(gender_data, use_container_width=True, hide_index=True)
    if not gender_data.empty:
        fig_gender = px.bar(gender_data, x="gender", y="dropout_count", color_discrete_sequence=["#026BC1"])
        fig_gender.update_layout(plot_bgcolor="white", height=350, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig_gender, use_container_width=True)