import streamlit as st
import sys
import os

# Go up one level for 'ui_utils' (app folder) and two levels for 'config' (root folder)
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

from config.database import fetch_data
from ui_utils import inject_enterprise_css

# CSS injection
inject_enterprise_css()

st.markdown("<h1>About - Student Dropout Prediction System</h1>", unsafe_allow_html=True)
st.markdown("This project implements an end-to-end student dropout prediction, dynamic exploratory data analysis, and multi-year forecasting system connected directly to the AP-RTGS data lake.")

st.markdown("### Core capabilities:")
st.markdown("* **Dynamic Impala Integration:** Ingests live data from the 168-column denormalized staging table, anchored on `person_id` to ensure all 10.14 million active students are evaluated.")
st.markdown("* **Advanced Feature Engineering:** Calculates historical stopout tracks, attendance momentum, and Haversine distance to school to build a highly predictive feature matrix.")
st.markdown("* **Native Categorical Modeling:** Trains modern tree-based algorithms (XGBoost/LightGBM) using a pandas category strategy to handle missing demographic data natively without destructive imputation.")
st.markdown("* **Real-time Explainability:** Exposes a FastAPI backend that scores students on-the-fly and returns SHAP-based feature impact explanations.")

st.markdown("### Why this matters:")
st.markdown("* Student dropout prediction is a **ranking and intervention problem** to allocate targeted state support resources effectively.")
st.markdown("* Shifting from Legacy Education IDs to Base Entity IDs (`person_id`) rescued over 1.7 million highly vulnerable students (such as fresh primary enrollments and higher education students) from being invisible to the ML system.")

st.markdown("### How to interpret the UI:")
st.markdown("* **Overview:** Choose an academic year to see aggregate dropout counts by district, class, and gender.")
st.markdown("* **Student Risk Scores:** Filter the student population by actual ML features, view real-time risk scores, and sort.")
st.markdown("* **Dashboard & Forecast:** Project future dropout volumes at the State, District, Class, or Gender level using growth/retention matrix projections.")
st.markdown("* **Top Risk Entities:** Identify exact administrative jurisdictions (districts/schools) requiring immediate intervention.")
st.markdown("* **Model Diagnostics:** Review the winning algorithm, confusion matrix performance on the 20% holdout set, and global feature importance.")
st.markdown("* **Drift Monitoring:** Track live statistical shifts in the predicted risk scores using dynamic database fetching.")
st.markdown("* **EDA:** View dynamic null-fill rates and the biological age risk curve directly from the Impala staging table.")

st.markdown("### Reproducible commands:")
st.markdown("1. Run Pipeline: `python ml_pipeline/train.py`")
st.markdown("2. Launch API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`")
st.markdown("3. Launch UI: `streamlit run app/app.py --server.address=0.0.0.0`")

st.markdown("<br><hr><p style='text-align: center; color: #64748B; font-size: 12px;'>AP-RTGS Data Science & Analytics</p>", unsafe_allow_html=True)

