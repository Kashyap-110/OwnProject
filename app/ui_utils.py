import streamlit as st

def inject_enterprise_css():
    """Injects custom CSS to give the Streamlit app a clean, enterprise look."""
    st.markdown("""
        <style>
        /* Base App Styling */
        .stApp { background-color: #F8FAFC; }
        [data-testid="stSidebar"] { background-color: #F1F5F9; border-right: 1px solid #E2E8F0; }
        
        /* Typography */
        h1, h2, h3, h4 { color: #1E293B; font-family: 'Inter', 'Segoe UI', sans-serif; font-weight: 700; }
        h1 { font-size: 2.2rem; margin-bottom: 0.2rem;}
        .subtitle-text { color: #64748B; font-size: 14px; margin-bottom: 25px; }
        
        /* KPI Cards */
        .kpi-container { 
            display: flex; 
            flex-direction: column; 
            background-color: #FFFFFF; 
            border-radius: 10px; 
            padding: 20px; 
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04); 
            border: 1px solid #E2E8F0; 
            margin-bottom: 20px; 
        }
        .kpi-title { color: #475569; font-size: 13px; font-weight: 600; margin-bottom: 8px;}
        .kpi-value { color: #0F172A; font-size: 36px; font-weight: 700; line-height: 1; margin: 0;}
        
        /* General Content Cards */
        .content-card { 
            background-color: #FFFFFF; 
            border-radius: 10px; 
            padding: 24px; 
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04); 
            border: 1px solid #E2E8F0; 
            margin-bottom: 24px; 
        }
        </style>
    """, unsafe_allow_html=True)

def create_kpi_card(title, value):
    """
    Helper function to dynamically generate HTML for a KPI card.
    """
    return f"""
    <div class="kpi-container">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """