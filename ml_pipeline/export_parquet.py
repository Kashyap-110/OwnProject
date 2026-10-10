import os
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.database import fetch_data

def export_to_parquet():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'raw_active_students.parquet')
    
    table_name = "aiml_usecase_curated.stage_std_prsn_hhld_schl_jvk_hlth_fcl_attn_marks_iti_ssdc_siblng_26sep"
    chunk_size = 500000
    last_id = None
    writer = None
    total_exported = 0
    schema = None
    
    print(f"Extracting active students from Impala to: {output_file}")
    start_time = time.time()
    
    while True:
        seek = f"AND CAST(person_id AS BIGINT) > {last_id}" if last_id else ""
        query = f"""
        SELECT * FROM {table_name}
        WHERE person_id IS NOT NULL AND LOWER(academic_26) = 'yes'
        {seek} 
        ORDER BY CAST(person_id AS BIGINT) ASC 
        LIMIT {chunk_size}
        """
        
        df = fetch_data(query)
        if df.empty: 
            break
            
        # 1. Force strict string casting for known messy text columns
        string_cols = ['hb_value', 'bmi', 'weight', 'vision', 'hearing', 'bp_value', 'identifier_value']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '')

        # 2. General numeric casting with strict float64 enforcement to prevent int64 vs double clashes
        for col in df.columns:
            if col not in string_cols and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
        
        # 3. Force explicit float64 on known numerical columns that switch between int/float
        float_cols = [
            'highest_qualification', 'class_currently_enrolled', 'student_id', 'student_pen', 
            'institution_id', 'student_district_id', 'student_mandal_id', 'secretariat_code',
            'domicile_status', 'guardian_person_id', 'household_income_category', 'school_id',
            'school_district_code', 'school_mandal_code', 'school_village_code', 'functional_status',
            'lowest_class', 'highest_class', 'total_students', 'total_boys', 'total_girls',
            'total_teachers', 'sanctioned_teachers', 'student_teacher_ratio', 'total_classrooms',
            'classrooms_good_condition', 'has_electricity', 'has_drinking_water', 'has_boys_toilet',
            'has_girls_toilet', 'has_cwsn_toilet', 'has_ramp', 'has_library', 'has_computer_lab',
            'has_smart_classroom', 'has_playground', 'has_boundary_wall', 'has_midday_meal',
            'internet_available', 'identifier_id', 'ub_code', 'course_group_code', 'ssdc_trainee_id',
            'ssdc_active', 'siblings_below_5'
        ]
        
        marks_cols = [c for c in df.columns if 'fa_' in c or 'sa_' in c or 'q1_' in c or 'q2_' in c or 'q3_' in c or 'q4_' in c or 'attendance' in c or 'working_days' in c]
        float_cols.extend(marks_cols)

        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

        # Create table and force schema matching
        table = pa.Table.from_pandas(df)
        
        if writer is None:
            # Capture the exact schema of the very first chunk
            schema = table.schema
            writer = pq.ParquetWriter(output_file, schema)
        else:
            # Force the current table to adopt the master schema from chunk 1
            table = table.cast(schema)
            
        writer.write_table(table)
        total_exported += len(df)
        last_id = int(df['person_id'].iloc[-1])
        
        print(f"Exported chunk... (Total saved: {total_exported:,})")

    if writer: 
        writer.close()
        
    elapsed = round((time.time() - start_time) / 60, 2)
    print(f"Successfully exported {total_exported:,} rows to Parquet in {elapsed} minutes.")

if __name__ == "__main__":
    export_to_parquet()
