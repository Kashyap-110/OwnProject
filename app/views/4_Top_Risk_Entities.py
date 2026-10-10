import streamlit as st
import pandas as pd
import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

from config.database import fetch_data
from ui_utils import inject_enterprise_css

inject_enterprise_css()

st.markdown("<h1>Top Risk Entities</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Top high-risk districts, schools, and classes</div>", unsafe_allow_html=True)

TABLE_NAME = "aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep"

@st.cache_data(ttl=3600)
def get_top_districts():
    query = f"""
    SELECT 
        CONCAT(COALESCE(student_district_name, 'Unknown'), ' (', CAST(COALESCE(student_district_id, 'N/A') AS STRING), ')') AS district, 
        COUNT(*) AS expected_dropouts
    FROM {TABLE_NAME}
    WHERE person_id IS NOT NULL
      AND LOWER(academic_25) = 'yes'
      AND (LOWER(academic_26) IN ('no', 'na', 'none', 'null', '') OR academic_26 IS NULL)
      AND COALESCE(ssdc_active, 0) != 1 
      AND UPPER(COALESCE(course_group_desc, '')) NOT LIKE '%ITI%' 
      AND UPPER(COALESCE(ub_name, '')) NOT LIKE '%ITI%' 
      AND class_currently_enrolled BETWEEN 1 AND 10
    GROUP BY CONCAT(COALESCE(student_district_name, 'Unknown'), ' (', CAST(COALESCE(student_district_id, 'N/A') AS STRING), ')')
    ORDER BY expected_dropouts DESC
    LIMIT 10
    """
    return fetch_data(query)

@st.cache_data(ttl=3600)
def get_top_schools():
    query = f"""
    SELECT 
        COALESCE(school_name, 'Unknown School') AS school_name, 
        COUNT(*) AS expected_dropouts
    FROM {TABLE_NAME}
    WHERE person_id IS NOT NULL
      AND LOWER(academic_25) = 'yes'
      AND (LOWER(academic_26) IN ('no', 'na', 'none', 'null', '') OR academic_26 IS NULL)
      AND COALESCE(ssdc_active, 0) != 1 
      AND UPPER(COALESCE(course_group_desc, '')) NOT LIKE '%ITI%' 
      AND UPPER(COALESCE(ub_name, '')) NOT LIKE '%ITI%' 
      AND class_currently_enrolled BETWEEN 1 AND 10
    GROUP BY COALESCE(school_name, 'Unknown School')
    ORDER BY expected_dropouts DESC
    LIMIT 10
    """
    return fetch_data(query)

@st.cache_data(ttl=3600)
def get_top_classes():
    query = f"""
    SELECT 
        class_currently_enrolled AS baseline_class, 
        COUNT(*) AS expected_dropouts
    FROM {TABLE_NAME}
    WHERE person_id IS NOT NULL
      AND LOWER(academic_25) = 'yes'
      AND (LOWER(academic_26) IN ('no', 'na', 'none', 'null', '') OR academic_26 IS NULL)
      AND COALESCE(ssdc_active, 0) != 1 
      AND UPPER(COALESCE(course_group_desc, '')) NOT LIKE '%ITI%' 
      AND UPPER(COALESCE(ub_name, '')) NOT LIKE '%ITI%' 
      AND class_currently_enrolled BETWEEN 1 AND 10
    GROUP BY class_currently_enrolled
    ORDER BY expected_dropouts DESC
    LIMIT 10
    """
    return fetch_data(query)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Top 10 High-Risk Districts")
    dist_data = get_top_districts()
    if not dist_data.empty:
        dist_data['expected_dropouts'] = dist_data['expected_dropouts'].astype(int)
        dist_data = dist_data.rename(columns={'district': 'District', 'expected_dropouts': 'Expected Dropouts'})
        st.dataframe(dist_data, use_container_width=True, hide_index=False)
    else:
        st.warning("No data available.")

with col2:
    st.markdown("### Top 10 High-Risk Schools")
    school_data = get_top_schools()
    if not school_data.empty:
        school_data['expected_dropouts'] = school_data['expected_dropouts'].astype(int)
        school_data = school_data.rename(columns={'school_name': 'School Name', 'expected_dropouts': 'Expected Dropouts'})
        st.dataframe(school_data, use_container_width=True, hide_index=False)
    else:
        st.warning("No data available.")

with col3:
    st.markdown("### Top 10 High-Risk Classes")
    class_data = get_top_classes()
    if not class_data.empty:
        class_data['baseline_class'] = class_data['baseline_class'].astype(str)
        class_data['expected_dropouts'] = class_data['expected_dropouts'].astype(int)
        class_data = class_data.rename(columns={'baseline_class': 'Class', 'expected_dropouts': 'Expected Dropouts'})
        st.dataframe(class_data, use_container_width=True, hide_index=False)
    else:
        st.warning("No data available.")