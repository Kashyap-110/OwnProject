import os
import sys
import time
import joblib
import json
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from config.database import fetch_data
from ml_pipeline.preprocess import clean_and_engineer_features, apply_manual_weight_matrix, calculate_terminal_transition_risk

def run_batch_scoring():
    print("?? Starting Enterprise Batch Scoring Job (Parquet Architecture)...")
    start_time = time.time()
    
    artifacts_dir = os.path.abspath(os.path.join(current_dir, '..', 'artifacts'))
    model_path = os.path.join(artifacts_dir, 'best_model.pkl')
    json_path = os.path.join(artifacts_dir, 'model_comparison.json')
    
    input_parquet_path = os.path.join(current_dir, '..', 'data', 'raw_active_students.parquet') 
    
    if not os.path.exists(model_path) or not os.path.exists(json_path):
        print("? Error: Model artifacts not found. Run train.py first.")
        return
        
    if not os.path.exists(input_parquet_path):
        print(f"? Error: Input Parquet file not found at {input_parquet_path}")
        return
        
    with open(json_path, 'r') as f:
        meta = json.load(f)
        
    print(f"Loading {meta.get('best_model_name', 'Best')} Model...")
    model = joblib.load(model_path)
    feature_names = meta.get("feature_names", [])
    
    print("Fetching dim_school master data for geospatial mapping...")
    TABLE_NAME = "aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep"
    school_query = f"SELECT DISTINCT school_id, school_category, school_type_coed_boys_girls, school_latitude AS latitude, school_longitude AS longitude FROM {TABLE_NAME} WHERE school_latitude IS NOT NULL"
    school_df = fetch_data(school_query)

    chunk_size = 250000 
    processed_count = 0
    all_scored_chunks = []
    
    parquet_file = pq.ParquetFile(input_parquet_path)
    
    print(f"\n?? Reading from Parquet: {input_parquet_path}")
    print(f"?? Total Rows to Process: {parquet_file.metadata.num_rows:,}")

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        raw_df = batch.to_pandas()
        processed_count += len(raw_df)
        
        print(f"\n?? Processing chunk... (Total processed: {processed_count:,})")
        
        raw_df['academic_26'] = raw_df['academic_26'].astype(str).str.lower()
        raw_df['class_currently_enrolled'] = pd.to_numeric(raw_df['class_currently_enrolled'], errors='coerce')
        
        # --- STRICT CLASS 1-10 FILTER FOR LIVE SCORING ---
        raw_df = raw_df[
            (raw_df['person_id'].notnull()) & 
            (raw_df['academic_26'] == 'yes') & 
            (raw_df['class_currently_enrolled'].between(1, 10))
        ].copy()
        
        if raw_df.empty:
            continue
            
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
        
        ui_df = raw_df[['person_id', 'school_name', 'baseline_class', 'student_gender', 
                        'student_district_name', 'social_category', 'income_band', 'dwelling_type']].copy()
        
        processed_df = clean_and_engineer_features(raw_df)
        processed_df = calculate_terminal_transition_risk(processed_df, school_df)
        
        for col in feature_names:
            if col not in processed_df.columns:
                processed_df[col] = 0
                
        X_live = processed_df[feature_names].copy()
        cat_columns = X_live.select_dtypes(include=['category']).columns
        for col in cat_columns:
            if col in X_live.columns:
                X_live[col] = X_live[col].astype('category')
            
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_live)[:, 1]
        else:
            probs = model.predict(X_live)
            
        ui_df['ml_risk_score'] = np.round(probs, 3)
        ui_df['ml_risk_level'] = np.where(ui_df['ml_risk_score'] >= 0.65, "High", np.where(ui_df['ml_risk_score'] >= 0.35, "Medium", "Low"))

        processed_df = apply_manual_weight_matrix(processed_df)
        ui_df['stakeholder_manual_score'] = processed_df['manual_risk_score']
        ui_df['stakeholder_risk_level'] = np.select(
            [ui_df['stakeholder_manual_score'] >= 81, ui_df['stakeholder_manual_score'] >= 61, ui_df['stakeholder_manual_score'] >= 31],
            ['Critical', 'High', 'Medium'], default='Low'
        )
        
        all_scored_chunks.append(ui_df)

    if all_scored_chunks:
        print("\n? Combining chunks and saving to Parquet Data Lake...")
        final_df = pd.concat(all_scored_chunks, ignore_index=True)
        output_file = os.path.join(artifacts_dir, 'statewide_scores.parquet')
        final_df.to_parquet(output_file, index=False)
        elapsed = round((time.time() - start_time) / 60, 2)
        print(f"?? Success! Scored {len(final_df):,} students in {elapsed} minutes.")
    else:
        print("?? No data was processed.")

if __name__ == "__main__":
    run_batch_scoring()
