import streamlit as st
import sys
import os

# Set page config at the very top
st.set_page_config(page_title="AP-RTGS | Overview", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state for authentication (None = not tried yet)
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = None

def login_screen():
    def password_entered():
        if st.session_state["password"] == "Admin@1234":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Clean up for security
        else:
            st.session_state["password_correct"] = False
            
    st.markdown("### Student Dropout System Login")
    st.text_input("Enter Access Password", type="password", on_change=password_entered, key="password")
    
    if st.session_state["password_correct"] is False:
        st.error("Incorrect Password. Please try again.")

# Define all application pages pointing to the NEW views/ directory
overview_page = st.Page("views/Overview.py", title="Overview", default=True)
risk_scores_page = st.Page("views/1_Student_Risk_Scores.py", title="Student Risk Scores")
dashboard_forecast_page = st.Page("views/2_Dashboard_and_Forecast.py", title="Dashboard & Forecast")
geospatial_page = st.Page("views/3_Geospatial_Analytics.py", title="Geospatial & Trends")
top_risk_page = st.Page("views/4_Top_Risk_Entities.py", title="Top Risk Entities")
diagnostics_page = st.Page("views/5_Model_Diagnostics.py", title="Model Diagnostics")
drift_page = st.Page("views/6_Drift_Monitoring.py", title="Drift Monitoring")
eda_page = st.Page("views/8_EDA.py", title="EDA")
ppt_page = st.Page("views/9_PPT_generator.py", title="Executive PPT Generator")
about_page = st.Page("views/7_About.py", title="About")

# Routing Logic: Dynamic Gatekeeper
if not st.session_state.get("password_correct", False):
    # Completely hide the sidebar UI
    st.markdown("""
        <style>
            [data-testid="collapsedControl"] { display: none; }
            [data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    
    # Register ONLY the login screen in the router. Other pages do not exist yet.
    login_page = st.Page(login_screen, title="Login")
    pg = st.navigation([login_page])
    
else:
    # Reveal the sidebar UI
    st.markdown("""
        <style>
            [data-testid="collapsedControl"] { display: flex; }
        </style>
    """, unsafe_allow_html=True)
    
    # Register the full application in the router
    pg = st.navigation({
        "AP-RTGS Student Dropout System": [
            overview_page, 
            risk_scores_page, 
            dashboard_forecast_page,
            geospatial_page,
            top_risk_page, 
            diagnostics_page, 
            drift_page,
            eda_page, 
            ppt_page,
            about_page
        ]
    })

# Execute the router
pg.run()
