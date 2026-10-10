from fastapi import FastAPI, HTTPException
import joblib
import json
import pandas as pd
import numpy as np
import sys
import os

from api.schemas import StudentPredictionRequest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.database import fetch_data
from ml_pipeline.preprocess import clean_and_engineer_features, apply_manual_weight_matrix, calculate_terminal_transition_risk

app = FastAPI(title="Explainable Student Churn API (Dual-Score)")

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
try:
    best_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'best_model.pkl'))
    explainer = joblib.load(os.path.join(ARTIFACTS_DIR, 'shap_explainer.pkl'))
    with open(os.path.join(ARTIFACTS_DIR, 'model_comparison.json'), 'r') as f:
        comparison_data = json.load(f)
except Exception as e:
    best_model, explainer, comparison_data = None, None, None
    print(f"Warning: Failed to load artifacts. Error: {e}")

try:
    print("Loading dim_school master data for API Geospatial lookups...")
    TABLE_NAME = "aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep"
    school_query = f"SELECT DISTINCT school_id, school_category, school_type_coed_boys_girls, school_latitude AS latitude, school_longitude AS longitude FROM {TABLE_NAME} WHERE school_latitude IS NOT NULL"
    school_df = fetch_data(school_query)
except Exception as e:
    school_df = pd.DataFrame()
    print(f"Warning: Failed to load school master data. {e}")

@app.get("/model-evaluation")
def get_model_evaluation():
    if not comparison_data:
        raise HTTPException(status_code=503, detail="Model evaluation artifacts missing.")
    return comparison_data

def get_manual_breakdown_dict(row):
    breakdown = []
    
    cls = 0 if pd.isna(row.get('baseline_class')) else row.get('baseline_class', 0)
    p_cls = 100 if cls in [5, 8, 9] else 40 if cls in [6, 7] else 0
    breakdown.append({"indicator_category": "Academic", "feature": "Baseline Class", "value": f"Class {int(cls)}", "points": p_cls * 0.0850})

    upper_limit = cls + 6
    age = upper_limit if pd.isna(row.get('student_age')) else row.get('student_age', upper_limit)
    gap = age - upper_limit
    p_gap = 100 if gap >= 3 else 60 if gap == 2 else 0
    breakdown.append({"indicator_category": "Academic", "feature": "Over-Age Gap", "value": f"Gap of {gap:.1f} Yrs", "points": p_gap * 0.0800})

    att = row.get('current_attendance', 100)
    att = 100 if pd.isna(att) else att
    p_att = 100 if att < 50 else 70 if att < 70 else 30 if att < 85 else 0
    breakdown.append({"indicator_category": "Behavioral", "feature": "Current Attendance", "value": f"{att:.1f}%", "points": p_att * 0.0700})

    fam = 100 if pd.isna(row.get('current_fa_marks')) else row.get('current_fa_marks', 100)
    p_fam = 100 if fam < 35 else 70 if fam < 50 else 30 if fam < 75 else 0
    breakdown.append({"indicator_category": "Academic", "feature": "Current FA Marks", "value": f"{fam:.1f}", "points": p_fam * 0.0650})

    sch = str(row.get('school_category', 'Unknown')).upper()
    p_sch = 100 if 'UNKNOWN' in sch else 5 if sch == 'UPPER PRIMARY ONLY' else 3 if 'PRIMARY' in sch else 0
    breakdown.append({"indicator_category": "Infrastructure", "feature": "School Category", "value": sch.title(), "points": p_sch * 0.0550})

    smt = 0 if pd.isna(row.get('sa_marks_trend')) else row.get('sa_marks_trend', 0)
    p_smt = 100 if smt <= -10 else 40 if smt <= -9 else 0
    breakdown.append({"indicator_category": "Behavioral", "feature": "SA Marks Trend", "value": f"{smt:.1f}", "points": p_smt * 0.0450})

    atr = 0 if pd.isna(row.get('attendance_trend')) else row.get('attendance_trend', 0)
    p_atr = 100 if atr <= -20 else 40 if atr <= -10 else 0
    breakdown.append({"indicator_category": "Behavioral", "feature": "Attendance Trend", "value": f"{atr:.1f}%", "points": p_atr * 0.0450})

    fmt = 0 if pd.isna(row.get('fa_marks_trend')) else row.get('fa_marks_trend', 0)
    p_fmt = 100 if fmt <= -10 else 40 if fmt <= -9 else 0
    breakdown.append({"indicator_category": "Behavioral", "feature": "FA Marks Trend", "value": f"{fmt:.1f}", "points": p_fmt * 0.0400})

    dst = 0 if pd.isna(row.get('distance_to_school_km')) else row.get('distance_to_school_km', 0)
    p_dst = 100 if dst > 10 else 75 if dst >= 6 else 40 if dst >= 3 else 0
    breakdown.append({"indicator_category": "Accessibility", "feature": "Dist to School", "value": f"{dst:.1f} km", "points": p_dst * 0.0350})

    tdr = 0 if pd.isna(row.get('teacher_deficit_ratio')) else row.get('teacher_deficit_ratio', 0)
    p_tdr = 100 if tdr >= 0.50 else 50 if tdr >= 0.25 else 0
    breakdown.append({"indicator_category": "Infrastructure", "feature": "Teacher Deficit", "value": f"{tdr*100:.0f}%", "points": p_tdr * 0.0350})

    med = str(row.get('medium_of_instruction', 'Unknown')).lower()
    p_med = 100 if med == 'tamil' else 40 if med == 'urdu' else 20 if med in ['english', 'telugu'] else 0
    breakdown.append({"indicator_category": "Social", "feature": "Medium of Instruction", "value": med.title(), "points": p_med * 0.0350})

    sam = 100 if pd.isna(row.get('current_sa_marks')) else row.get('current_sa_marks', 100)
    p_sam = 100 if sam < 35 else 70 if sam < 50 else 30 if sam < 75 else 0
    breakdown.append({"indicator_category": "Academic", "feature": "Current SA Marks", "value": f"{sam:.1f}", "points": p_sam * 0.0300})

    par = str(row.get('parental_presence', 'Unknown'))
    p_par = 0 if par == 'Both IDs Present' else 50 if par in ['Mother ID Only', 'Father ID Only'] else 100
    breakdown.append({"indicator_category": "Socio-Economic", "feature": "Parental Presence", "value": par, "points": p_par * 0.0300})

    tv = str(row.get('current_thalliki_vandanam_status', 'No')).lower()
    p_tv = 0 if tv == 'yes' else 100
    breakdown.append({"indicator_category": "Welfare", "feature": "TV Status", "value": tv.capitalize(), "points": p_tv * 0.0300})

    dwl = str(row.get('dwelling_type', 'Unknown')).lower()
    p_dwl = 100 if 'flat' in dwl or dwl == 'unknown' else 39 if 'duplex' in dwl else 10 if 'kaccha' in dwl else 0
    breakdown.append({"indicator_category": "Socio-Economic", "feature": "Dwelling Type", "value": dwl.title(), "points": p_dwl * 0.0250})

    sib = 0 if pd.isna(row.get('siblings_below_5')) else row.get('siblings_below_5', 0)
    p_sib = 100 if sib >= 2 else 50 if sib == 1 else 0
    breakdown.append({"indicator_category": "Family", "feature": "Siblings Below 5 Yrs", "value": str(int(sib)), "points": p_sib * 0.0200})

    jvk = 4 if pd.isna(row.get('student_kit_score')) else row.get('student_kit_score', 4)
    p_jvk = 100 if jvk == 0 else 50 if jvk <= 2 else 0
    breakdown.append({"indicator_category": "Welfare", "feature": "Student Kit", "value": f"{int(jvk)}/4 Items", "points": p_jvk * 0.0200})

    d_next = 0 if pd.isna(row.get('distance_to_next_school_km')) else row.get('distance_to_next_school_km', 0)
    p_dnext = 100 if d_next > 10 else 75 if d_next >= 6 else 40 if d_next >= 3 else 0
    breakdown.append({"indicator_category": "Geospatial", "feature": "Dist to Next School", "value": f"{d_next:.1f} km", "points": p_dnext * 0.0200})

    soc = str(row.get('social_category', 'Unknown')).upper()
    p_soc = 0 if soc == 'SC' else 7 if soc == 'BC' else 24 if soc == 'ST' else 45 if soc == 'OC' else 100
    breakdown.append({"indicator_category": "Social", "feature": "Social Category", "value": soc, "points": p_soc * 0.0150})

    mar = str(row.get('marital_status', 'Unknown')).lower()
    p_mar = 0 if mar in ['unmarried', 'unknown'] else 100
    breakdown.append({"indicator_category": "Social", "feature": "Marital Status", "value": mar.title(), "points": p_mar * 0.0150})

    hv = 0 if pd.isna(row.get('health_vulnerability_score')) else row.get('health_vulnerability_score', 0)
    p_hv = 100 if hv >= 3 else 70 if hv == 2 else 30 if hv == 1 else 0
    breakdown.append({"indicator_category": "Health", "feature": "Health Vulnerability", "value": f"Score {int(hv)}", "points": p_hv * 0.0150})

    mem = 4 if pd.isna(row.get('members_count')) else row.get('members_count', 4)
    p_mem = 100 if mem == 1 or mem >= 15 else 50 if 9 <= mem <= 14 else 0
    breakdown.append({"indicator_category": "Socio-Economic", "feature": "Members Count", "value": f"{int(mem)}", "points": p_mem * 0.0150})

    inv = 0 if pd.isna(row.get('age_invalid_flag')) else row.get('age_invalid_flag', 0)
    p_inv = 100 if inv == 1 else 0
    breakdown.append({"indicator_category": "System", "feature": "Age Invalid", "value": "Yes" if inv == 1 else "No", "points": p_inv * 0.0150})

    is_fem = str(row.get('student_gender', '')).lower() == 'female'
    d_girl = 0 if pd.isna(row.get('nearest_distance_to_girl_school')) else row.get('nearest_distance_to_girl_school', 0)
    p_girl = 100 if (is_fem and d_girl > 3) else 0
    breakdown.append({"indicator_category": "Accessibility", "feature": "Nearest Girl School", "value": f"{d_girl:.1f} km", "points": p_girl * 0.0100})

    water = 1 if pd.isna(row.get('has_drinking_water')) else row.get('has_drinking_water', 1)
    p_water = 100 if water == 0 else 0
    breakdown.append({"indicator_category": "Infrastructure", "feature": "Drinking Water", "value": "Yes" if water == 1 else "No", "points": p_water * 0.0100})

    rat = str(row.get('ration_card_type', 'Unknown')).upper()
    p_rat = 0 if rat in ['AAY', 'BPL'] else 100
    breakdown.append({"indicator_category": "Socio-Economic", "feature": "Ration Card", "value": rat, "points": p_rat * 0.0100})

    ru = str(row.get('rural_urban_flag', 'Unknown')).upper()
    p_ru = 0 if ru == 'R' else 100
    breakdown.append({"indicator_category": "Accessibility", "feature": "Rural/Urban", "value": ru, "points": p_ru * 0.0100})

    trb = 0 if pd.isna(row.get('is_tribal')) else row.get('is_tribal', 0)
    p_trb = 100 if trb == 1 else 0
    breakdown.append({"indicator_category": "Geospatial", "feature": "Tribal Mandal", "value": "Yes" if trb == 1 else "No", "points": p_trb * 0.0100})

    inc = str(row.get('income_band', 'Unknown')).replace(' ', '')
    p_inc = 0 if inc == '>5L' else 10 if inc == '3L-5L' else 25 if inc == '0-1L' else 33 if inc == '1L-3L' else 100
    breakdown.append({"indicator_category": "Socio-Economic", "feature": "Income Band", "value": inc, "points": p_inc * 0.0100})

    g_toilet = 1 if pd.isna(row.get('has_girls_toilet')) else row.get('has_girls_toilet', 1)
    p_toilet = 100 if (is_fem and g_toilet == 0) else 0
    breakdown.append({"indicator_category": "Infrastructure", "feature": "Girls Toilet", "value": "Yes" if g_toilet == 1 else "No", "points": p_toilet * 0.0100})

    gen = str(row.get('student_gender', 'Unknown')).lower()
    p_gen = 0 if gen == 'female' else 4 if gen == 'male' else 31 if gen == 'transgender' else 100
    breakdown.append({"indicator_category": "Social", "feature": "Gender", "value": gen.title(), "points": p_gen * 0.0100})

    rej = 0 if pd.isna(row.get('is_recent_rejoiner')) else row.get('is_recent_rejoiner', 0)
    p_rej = 100 if rej == 1 else 0
    breakdown.append({"indicator_category": "Geospatial", "feature": "Recent Rejoiner", "value": "Yes" if rej == 1 else "No", "points": p_rej * 0.0100})

    hmy = 0 if pd.isna(row.get('historical_missing_years')) else row.get('historical_missing_years', 0)
    p_hmy = 100 if hmy >= 2 else 60 if hmy == 1 else 0
    breakdown.append({"indicator_category": "Profile", "feature": "Missing Years", "value": int(hmy), "points": p_hmy * 0.0100})

    t_risk = 0 if pd.isna(row.get('is_terminal_transition_risk')) else row.get('is_terminal_transition_risk', 0)
    p_trisk = 100 if t_risk == 1 else 0
    breakdown.append({"indicator_category": "Special Needs", "feature": "Terminal Transition Risk", "value": "Yes" if t_risk == 1 else "No", "points": p_trisk * 0.0100})

    req_acc = 0 if pd.isna(row.get('requires_accessible_school')) else row.get('requires_accessible_school', 0)
    p_req = 100 if req_acc == 1 else 0
    breakdown.append({"indicator_category": "Special Needs", "feature": "Requires Accessible School", "value": "Yes" if req_acc == 1 else "No", "points": p_req * 0.0500})

    return [b for b in breakdown if b['points'] > 0]

@app.post("/predict")
def predict_with_explanation(student: StudentPredictionRequest):
    if not best_model or not explainer:
        raise HTTPException(status_code=503, detail="Prediction models not loaded.")
        
    raw_data = student.dict()
    df = pd.DataFrame([raw_data])
    
    df = clean_and_engineer_features(df)
    
    if not school_df.empty:
        df = calculate_terminal_transition_risk(df, school_df)
    else:
        df['distance_to_next_school_km'] = np.nan 
        df['nearest_distance_to_girl_school'] = np.nan
        df['is_terminal_transition_risk'] = 0
    
    df = apply_manual_weight_matrix(df)
    manual_score = float(df['manual_risk_score'].iloc[0])
    
    if manual_score >= 81:
        manual_tier = "Critical"
    elif manual_score >= 61:
        manual_tier = "High"
    elif manual_score >= 31:
        manual_tier = "Medium"
    else:
        manual_tier = "Low"
        
    manual_explanations = get_manual_breakdown_dict(df.iloc[0].to_dict())
    
    feature_names = comparison_data.get('feature_names', [])
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
            
    input_df = df[feature_names].copy()
    categorical_options = comparison_data.get('categorical_options', {})
    for col in categorical_options.keys():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).replace(['nan', 'None', ''], np.nan).astype('category')
            
    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(input_df)[0][1]
    else:
        prob = best_model.predict(input_df)[0]
        
    ml_score_pct = round(prob * 100, 2)
    ml_tier = "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low"
    
    if hasattr(best_model, 'named_steps'):
        processed_input = best_model.named_steps['preprocessor'].transform(input_df)
        shap_values = explainer.shap_values(processed_input)
    else:
        try:
            shap_values = explainer(input_df)
        except Exception as e:
            # --- FALLBACK: Translating categorical strings to numeric encodings for Scikit-Learn ---
            if "could not convert string to float" in str(e):
                input_shap = input_df.copy()
                for col in categorical_options.keys():
                    if col in input_shap.columns:
                        input_shap[col] = pd.Categorical(input_shap[col], categories=categorical_options[col])
                        input_shap[col] = input_shap[col].cat.codes
                shap_values = explainer(input_shap)
            else:
                raise e
        
    vals = shap_values.values[0, :, 1] if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3 else shap_values[0] if isinstance(shap_values, list) else shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0]
    
    if len(vals.shape) > 1:
        vals = vals[:, 1]

    abs_vals = np.abs(vals)
    total_impact = np.sum(abs_vals)
    
    ai_explanations = []
    for feat, val, abs_v in zip(feature_names, vals, abs_vals):
        pct_contrib = round((abs_v / total_impact) * 100, 2) if total_impact > 0 else 0
        display_val = raw_data.get(feat, input_df[feat].iloc[0])
        
        ai_explanations.append({
            "feature": feat, 
            "input_value": str(display_val),
            "impact_direction": "Increases Risk" if val > 0 else "Decreases Risk",
            "contribution_percentage": pct_contrib
        })
        
    return {
        "ml_risk_score": ml_score_pct,
        "ml_risk_tier": ml_tier,
        "stakeholder_manual_score": manual_score,
        "stakeholder_manual_tier": manual_tier,
        "tier_match": ml_tier == manual_tier,
        "best_model_used": comparison_data.get('best_model_name', 'Unknown'),
        "ai_shap_explanations": sorted(ai_explanations, key=lambda x: x['contribution_percentage'], reverse=True),
        "manual_matrix_explanations": sorted(manual_explanations, key=lambda x: x['points'], reverse=True)
    }