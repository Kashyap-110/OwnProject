import joblib
import json
import pandas as pd
import numpy as np
import os
import sys
import time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.database import fetch_data
from ml_pipeline.preprocess import clean_and_engineer_features, calculate_terminal_transition_risk

TABLE_NAME = "aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep"

def train_and_evaluate_models():
    print("Fetching training data via 4 Rolling Windows (Pure 5% Random Downsampling)...")
    start_time = time.time()
    
    query = f"""
    WITH stacked_windows AS (
        SELECT 
            person_id, 
            (YEAR(CAST('2025-06-01' AS DATE)) - YEAR(CAST(student_date_of_birth AS DATE))) AS student_age, 
            student_gender, social_category, income_band, 
            dwelling_type, ration_card_type, father_person_id, mother_person_id, 
            school_category, thalliki_vandanam_25 AS current_thalliki_vandanam_status, rural_urban_flag, members_count,
            marital_status, medium_of_instruction, siblings_below_5,
            student_district_name, student_mandal_name,
            school_management, school_type_coed_boys_girls,
            is_anemic, child_condition, fourd_condition, hb_status,
            house_latitude, house_longitude, school_latitude, school_longitude, 
            academic_22, academic_23, academic_24, academic_25, academic_26,
            ssdc_active, course_group_desc, ub_name,
            uniform_rec, textbooks_rec, bags_rec, shoes_rec,
            sanctioned_teachers, total_teachers,
            has_drinking_water, has_girls_toilet, has_electricity,
            '2025' AS baseline_year,
            class_currently_enrolled AS baseline_class,
            LOWER(academic_25) AS baseline_status,
            LOWER(academic_26) AS target_status,
            LOWER(academic_24) AS historical_status_1,
            LOWER(academic_23) AS historical_status_2,
            LOWER(academic_22) AS historical_status_3,
            attendance_percentage_2024_2025 AS current_attendance,
            attendance_percentage_2023_2024 AS previous_attendance,
            fa_2024_2025 AS current_fa_marks,
            fa_2023_2024 AS previous_fa_marks,
            sa_2024_2025 AS current_sa_marks,
            sa_2023_2024 AS previous_sa_marks
        FROM {TABLE_NAME}
        WHERE LOWER(academic_25) = 'yes'

        UNION ALL
        SELECT 
            person_id, 
            (YEAR(CAST('2024-06-01' AS DATE)) - YEAR(CAST(student_date_of_birth AS DATE))) AS student_age, 
            student_gender, social_category, income_band, 
            dwelling_type, ration_card_type, father_person_id, mother_person_id, 
            school_category, thalliki_vandanam_24 AS current_thalliki_vandanam_status, rural_urban_flag, members_count,
            marital_status, medium_of_instruction, siblings_below_5,
            student_district_name, student_mandal_name,
            school_management, school_type_coed_boys_girls,
            is_anemic, child_condition, fourd_condition, hb_status,
            house_latitude, house_longitude, school_latitude, school_longitude, 
            academic_22, academic_23, academic_24, academic_25, academic_26,
            ssdc_active, course_group_desc, ub_name,
            uniform_rec, textbooks_rec, bags_rec, shoes_rec,
            sanctioned_teachers, total_teachers,
            has_drinking_water, has_girls_toilet, has_electricity,
            '2024' AS baseline_year,
            class_currently_enrolled AS baseline_class,
            LOWER(academic_24) AS baseline_status,
            LOWER(academic_25) AS target_status,
            LOWER(academic_23) AS historical_status_1,
            LOWER(academic_22) AS historical_status_2,
            NULL AS historical_status_3,
            attendance_percentage_2023_2024 AS current_attendance,
            attendance_percentage_2022_2023 AS previous_attendance,
            fa_2023_2024 AS current_fa_marks,
            fa_2022_2023 AS previous_fa_marks,
            sa_2023_2024 AS current_sa_marks,
            sa_2022_2023 AS previous_sa_marks
        FROM {TABLE_NAME}
        WHERE LOWER(academic_24) = 'yes'

        UNION ALL
        SELECT 
            person_id, 
            (YEAR(CAST('2023-06-01' AS DATE)) - YEAR(CAST(student_date_of_birth AS DATE))) AS student_age, 
            student_gender, social_category, income_band, 
            dwelling_type, ration_card_type, father_person_id, mother_person_id, 
            school_category, thalliki_vandanam_23 AS current_thalliki_vandanam_status, rural_urban_flag, members_count,
            marital_status, medium_of_instruction, siblings_below_5,
            student_district_name, student_mandal_name,
            school_management, school_type_coed_boys_girls,
            is_anemic, child_condition, fourd_condition, hb_status,
            house_latitude, house_longitude, school_latitude, school_longitude, 
            academic_22, academic_23, academic_24, academic_25, academic_26,
            ssdc_active, course_group_desc, ub_name,
            uniform_rec, textbooks_rec, bags_rec, shoes_rec,
            sanctioned_teachers, total_teachers,
            has_drinking_water, has_girls_toilet, has_electricity,
            '2023' AS baseline_year,
            class_currently_enrolled AS baseline_class,
            LOWER(academic_23) AS baseline_status,
            LOWER(academic_24) AS target_status,
            LOWER(academic_22) AS historical_status_1,
            NULL AS historical_status_2,
            NULL AS historical_status_3,
            attendance_percentage_2022_2023 AS current_attendance,
            NULL AS previous_attendance, 
            fa_2022_2023 AS current_fa_marks,
            NULL AS previous_fa_marks,
            sa_2022_2023 AS current_sa_marks,
            NULL AS previous_sa_marks
        FROM {TABLE_NAME}
        WHERE LOWER(academic_23) = 'yes'
        
        UNION ALL
        SELECT 
            person_id, 
            (YEAR(CAST('2022-06-01' AS DATE)) - YEAR(CAST(student_date_of_birth AS DATE))) AS student_age, 
            student_gender, social_category, income_band, 
            dwelling_type, ration_card_type, father_person_id, mother_person_id, 
            school_category, thalliki_vandanam_22 AS current_thalliki_vandanam_status, rural_urban_flag, members_count,
            marital_status, medium_of_instruction, siblings_below_5,
            student_district_name, student_mandal_name,
            school_management, school_type_coed_boys_girls,
            is_anemic, child_condition, fourd_condition, hb_status,
            house_latitude, house_longitude, school_latitude, school_longitude, 
            academic_22, academic_23, academic_24, academic_25, academic_26,
            ssdc_active, course_group_desc, ub_name,
            uniform_rec, textbooks_rec, bags_rec, shoes_rec,
            sanctioned_teachers, total_teachers,
            has_drinking_water, has_girls_toilet, has_electricity,
            '2022' AS baseline_year,
            class_currently_enrolled AS baseline_class,
            LOWER(academic_22) AS baseline_status,
            LOWER(academic_23) AS target_status,
            NULL AS historical_status_1,
            NULL AS historical_status_2,
            NULL AS historical_status_3,
            NULL AS current_attendance, 
            NULL AS previous_attendance, 
            NULL AS current_fa_marks,
            NULL AS previous_fa_marks,
            NULL AS current_sa_marks,
            NULL AS previous_sa_marks
        FROM {TABLE_NAME}
        WHERE LOWER(academic_22) = 'yes'
    )
    
    SELECT * FROM stacked_windows
    WHERE person_id IS NOT NULL 
      AND baseline_class BETWEEN 1 AND 10 
      AND (
          (
            (target_status IN ('no', 'na', 'none', 'null', '') OR target_status IS NULL) 
            AND COALESCE(ssdc_active, 0) != 1 
            AND UPPER(COALESCE(course_group_desc, '')) NOT LIKE '%ITI%' 
            AND UPPER(COALESCE(ub_name, '')) NOT LIKE '%ITI%'
          )
          OR 
          (target_status IN ('yes', 'y') AND ABS(fnv_hash(CAST(person_id AS STRING))) % 100 <= 5)
      )
    """
    
    raw_df = fetch_data(query)
    
    if raw_df.empty:
        print("No data fetched from DB. Check Impala connection.")
        return
        
    print(f"Fetch complete in {round((time.time() - start_time), 2)} seconds. Total Rows: {len(raw_df):,}")
    
    print("Fetching dim_school master data for geospatial mapping...")
    school_query = f"SELECT DISTINCT school_id, school_category, school_type_coed_boys_girls, school_latitude AS latitude, school_longitude AS longitude FROM {TABLE_NAME} WHERE school_latitude IS NOT NULL"
    school_df = fetch_data(school_query)
        
    print("Executing Preprocessing and Feature Engineering...")
    df = clean_and_engineer_features(raw_df)
    df = calculate_terminal_transition_risk(df, school_df)
    
    features = [
        'student_age', 'student_gender', 'social_category', 'income_band', 
        'dwelling_type', 'ration_card_type', 'members_count', 'current_thalliki_vandanam_status',
        'rural_urban_flag', 'school_category', 'baseline_class', 
        'current_attendance', 'attendance_trend', 
        'current_fa_marks', 'fa_marks_trend', 
        'current_sa_marks', 'sa_marks_trend',
        'marital_status', 'medium_of_instruction', 'siblings_below_5',
        'distance_to_school_km', 'parental_presence', 'age_invalid_flag', 
        'historical_missing_years', 'is_recent_rejoiner',
        'distance_to_next_school_km', 'is_terminal_transition_risk',
        'is_tribal', 'nearest_distance_to_girl_school', 'requires_accessible_school', 
        'health_vulnerability_score', 'student_kit_score', 
        'teacher_deficit_ratio', 'has_drinking_water', 'has_girls_toilet'
    ]
    
    features = [f for f in features if f in df.columns]
    
    X = df[features + ['person_id']].copy() 
    y = df['is_churned']
    
    print("Enforcing strictly typed schema for Machine Learning matrices...")
    categorical_cols = ['student_gender', 'social_category', 'income_band', 'dwelling_type', 
                        'ration_card_type', 'parental_presence', 'school_category', 
                        'current_thalliki_vandanam_status', 'rural_urban_flag', 'marital_status', 'medium_of_instruction']
                        
    for col in features:
        if col in categorical_cols:
            X[col] = X[col].astype(str).replace(['nan', 'None', '', 'null'], 'Unknown').astype('category')
        elif col in ['has_drinking_water', 'has_girls_toilet']:
            X[col] = X[col].astype(str).str.lower().map({'true': 1, 'yes': 1, '1': 1, '1.0': 1}).fillna(0).astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    categorical_options = {}
    cat_columns = [col for col in X.select_dtypes(include=['category']).columns.tolist() if col != 'person_id']
    for col in cat_columns:
        valid_cats = [str(c) for c in X[col].cat.categories if pd.notnull(c)]
        categorical_options[col] = valid_cats
        
    print("Splitting Data by Person ID (80% Train, 20% Test) to prevent leakage...")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=X['person_id']))
    
    X_train = X.iloc[train_idx].drop(columns=['person_id'])
    X_test = X.iloc[test_idx].drop(columns=['person_id'])
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = round(neg_count / pos_count, 2) if pos_count > 0 else 1.0
    print(f"New Data Shape -> Active: {neg_count:,} | Dropouts: {pos_count:,}")
    print(f"Applying scale_pos_weight: {spw}")
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
    
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), categorical_features)
        ])
        
    linear_preprocessor = Pipeline([
        ('base', base_preprocessor),
        ('scaler', StandardScaler())
    ])
    
    candidate_models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            random_state=42, enable_categorical=True, tree_method='hist',
            scale_pos_weight=spw, colsample_bytree=0.7
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            random_state=42, verbose=-1, scale_pos_weight=spw, colsample_bytree=0.7
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1, 
            random_state=42, verbose=0, auto_class_weights='Balanced',
            cat_features=categorical_features
        ),
        "Scikit_HistGBM": HistGradientBoostingClassifier(
            max_iter=100, max_depth=6, learning_rate=0.1, 
            random_state=42, categorical_features=categorical_features,
            class_weight='balanced'
        )
    }
    
    comparison_results = {}
    best_score = -1
    best_model_name = ""
    best_model_obj = None
    
    for name, model in candidate_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1] 
        else:
            probs = preds
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        params = model.get_params()
        params_serializable = {k: str(v) for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
        
        comparison_results[name] = {
            "metrics": {
                "Accuracy": round(acc, 4), 
                "Precision": round(prec, 4), 
                "Recall": round(rec, 4), 
                "F1 Score": round(f1, 4), 
                "ROC AUC": round(auc, 4),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            },
            "parameters": params_serializable
        }
        
        if auc > best_score:
            best_score = auc
            best_model_name = name
            best_model_obj = model
            
    print(f"Best Model: {best_model_name} (AUC: {best_score})")
            
    base_dir = os.path.dirname(__file__)
    artifacts_dir = os.path.abspath(os.path.join(base_dir, '..', 'artifacts'))
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print("Saving Models and Metadata...")
    joblib.dump(best_model_obj, os.path.join(artifacts_dir, 'best_model.pkl'))
    with open(os.path.join(artifacts_dir, 'model_comparison.json'), 'w') as f:
        json.dump({
            "best_model_name": best_model_name,
            "comparison": comparison_results,
            "feature_names": features,
            "categorical_options": categorical_options
        }, f, indent=4)
        
    print("Generating SHAP Explainer...")
    explainer = shap.TreeExplainer(best_model_obj)
    joblib.dump(explainer, os.path.join(artifacts_dir, 'shap_explainer.pkl'))
    
    print("Exporting data splits for UI diagnostics...")
    X_train.to_csv(os.path.join(artifacts_dir, 'X_train_sample.csv'), index=False)
    X_test.to_csv(os.path.join(artifacts_dir, 'X_test_sample.csv'), index=False)
    
    print("Pipeline Complete.")

if __name__ == "__main__":
    train_and_evaluate_models()