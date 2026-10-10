import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# Hardcoded set for rapid offline matching in production environments
TRIBAL_MANDALS = {
    ('ALLURI SITHARAMA RAJU', 'ANANTHAGIRI'), ('ALLURI SITHARAMA RAJU', 'ARAKUVALLEY'), ('ALLURI SITHARAMA RAJU', 'CHINTAPALLI'),
    ('ALLURI SITHARAMA RAJU', 'DUMBRIGUDA'), ('ALLURI SITHARAMA RAJU', 'G.K.VEEDHI'), ('ALLURI SITHARAMA RAJU', 'G.MADUGULA'),
    ('ALLURI SITHARAMA RAJU', 'HUKUMPETA'), ('ALLURI SITHARAMA RAJU', 'KOYYURU'), ('ALLURI SITHARAMA RAJU', 'MUNCHINGPUT'),
    ('ALLURI SITHARAMA RAJU', 'PADERU'), ('ALLURI SITHARAMA RAJU', 'PEDABAYALU'), ('ANAKAPALLI', 'CHEEDIKADA'),
    ('ANAKAPALLI', 'DEVARAPALLI'), ('ANAKAPALLI', 'GOLUGONDA'), ('ANAKAPALLI', 'NARASIPATNAM'), ('ANAKAPALLI', 'NATHAVARAM'),
    ('ANAKAPALLI', 'RAVIKAMATHAM'), ('ANAKAPALLI', 'ROLUGUNTA'), ('ANAKAPALLI', 'V.MADUGULA'), ('ANNAMAYYA', 'CHINNAMANDEM'),
    ('ANNAMAYYA', 'GALIVEEDU'), ('ANNAMAYYA', 'L.R.PALLI'), ('ANNAMAYYA', 'RAMAPURAM'), ('ANNAMAYYA', 'RAYACHOTY'),
    ('ANNAMAYYA', 'SAMBEPALLI'), ('EAST GODAVARI', 'GOKAVARAM'), ('EAST GODAVARI', 'KORUKONDA'), ('EAST GODAVARI', 'SEETHANAGARAM'),
    ('ELURU', 'BUTTAIGUDEM'), ('ELURU', 'DWARAKATIRUMALA'), ('ELURU', 'JANGAREDDYGUDEM'), ('ELURU', 'JEELUGUMILLI'),
    ('ELURU', 'KAMAVARAPUKOTA'), ('ELURU', 'KOYYALAGUDEM'), ('ELURU', 'KUKUNOOR'), ('ELURU', 'POLAVARAM'),
    ('ELURU', 'T.NARSAPURAM'), ('ELURU', 'VELAIRPADU'), ('MANYAM', 'BALIJIPETA'), ('MANYAM', 'BHAMINI'),
    ('MANYAM', 'G.L.PURAM'), ('MANYAM', 'GARUGUBILLI'), ('MANYAM', 'JIYYAMMAVALASA'), ('MANYAM', 'KOMARADA'),
    ('MANYAM', 'KURUPAM'), ('MANYAM', 'MAKKUVA'), ('MANYAM', 'PACHIPENTA'), ('MANYAM', 'PALAKONDA'),
    ('MANYAM', 'PARVATHIPURAM'), ('MANYAM', 'SALURU'), ('MANYAM', 'SEETHAMPETA'), ('MANYAM', 'SEETHANAGARAM'),
    ('MANYAM', 'VEERAGHATTAM'), ('NANDYAL', 'SRISAILAM'), ('POLAVARAM', 'ADDATEEGALA'), ('POLAVARAM', 'CHINTUR'),
    ('POLAVARAM', 'DEVIPATNAM'), ('POLAVARAM', 'GANGAVARAM'), ('POLAVARAM', 'GURTHEDU'), ('POLAVARAM', 'KUNAVARAM'),
    ('POLAVARAM', 'MAREDUMILLI'), ('POLAVARAM', 'NELLIPAKA'), ('POLAVARAM', 'RAJAVOMMANGI'), ('POLAVARAM', 'RAMPACHODAVARAM'),
    ('POLAVARAM', 'V.R.PURAM'), ('POLAVARAM', 'Y.RAMAVARAM'), ('SRIKAKULAM', 'BURJA'), ('SRIKAKULAM', 'G.SIGADAM'),
    ('SRIKAKULAM', 'HIRAMANDALAM'), ('SRIKAKULAM', 'JALUMURU'), ('SRIKAKULAM', 'KOTTURU'), ('SRIKAKULAM', 'LAVERU'),
    ('SRIKAKULAM', 'MELIAPUTTI'), ('SRIKAKULAM', 'NANDIGAM'), ('SRIKAKULAM', 'NARASANNAPETA'), ('SRIKAKULAM', 'PATHAPATNAM'),
    ('SRIKAKULAM', 'POLAKI'), ('SRIKAKULAM', 'SARAVAKOTA'), ('SRIKAKULAM', 'SARUBUJJILI'), ('VIZIANAGARAM', 'BADANGI'),
    ('VIZIANAGARAM', 'BONDAPALLI'), ('VIZIANAGARAM', 'DATHIRAJERU'), ('VIZIANAGARAM', 'GARIVIDI'), ('VIZIANAGARAM', 'MENTADA'),
    ('VIZIANAGARAM', 'MERAKAMUDIDAM'), ('VIZIANAGARAM', 'R.AMADALAVALASA'), ('VIZIANAGARAM', 'RAMABHADRAPURAM'),
    ('VIZIANAGARAM', 'SANTHAKAVITI'), ('VIZIANAGARAM', 'THERLAM'), ('VIZIANAGARAM', 'VANGARA')
}

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_terminal_transition_risk(student_df: pd.DataFrame, school_df: pd.DataFrame) -> pd.DataFrame:
    print("--> Executing Spatial BallTree for Transitions and Girl-Specific Schools...")
    
    if not school_df.empty:
        school_df['latitude'] = pd.to_numeric(school_df['latitude'], errors='coerce')
        school_df['longitude'] = pd.to_numeric(school_df['longitude'], errors='coerce')
        school_df = school_df.dropna(subset=['latitude', 'longitude'])
        if 'school_type_coed_boys_girls' not in school_df.columns:
            school_df['school_type_coed_boys_girls'] = 'CO_ED'
    
    up_schools = school_df[~school_df['school_category'].astype(str).str.upper().isin(['PRIMARY', 'PRIMARY ONLY', 'NAN', 'UNKNOWN'])].copy()
    sec_schools = school_df[school_df['school_category'].astype(str).str.contains('SECONDARY|HIGH', case=False, na=False)].copy()
    
    girls_up_schools = up_schools[up_schools['school_type_coed_boys_girls'].astype(str).str.upper().isin(['GIRLS', 'CO_ED', 'CO-ED', 'NAN'])]
    girls_sec_schools = sec_schools[sec_schools['school_type_coed_boys_girls'].astype(str).str.upper().isin(['GIRLS', 'CO_ED', 'CO-ED', 'NAN'])]
    
    student_df['house_latitude'] = pd.to_numeric(student_df['house_latitude'], errors='coerce')
    student_df['house_longitude'] = pd.to_numeric(student_df['house_longitude'], errors='coerce')
    
    valid_coords = student_df['house_latitude'].notnull() & student_df['house_longitude'].notnull()
    is_female = student_df.get('student_gender', pd.Series(['Male']*len(student_df))).astype(str).str.upper() == 'FEMALE'
    valid_female_coords = is_female & valid_coords
    
    student_df['distance_to_next_school_km'] = np.nan
    student_df['nearest_distance_to_girl_school'] = np.nan
    student_df['is_terminal_transition_risk'] = 0
    
    if valid_coords.any() and not up_schools.empty and not sec_schools.empty:
        student_coords = np.radians(student_df.loc[valid_coords, ['house_latitude', 'house_longitude']].values)
        
        tree_up = BallTree(np.radians(up_schools[['latitude', 'longitude']].values), metric='haversine')
        dist_up, _ = tree_up.query(student_coords, k=1)
        
        tree_sec = BallTree(np.radians(sec_schools[['latitude', 'longitude']].values), metric='haversine')
        dist_sec, _ = tree_sec.query(student_coords, k=1)
        
        student_df.loc[valid_coords, 'dist_up_km'] = dist_up.flatten() * 6371.0
        student_df.loc[valid_coords, 'dist_sec_km'] = dist_sec.flatten() * 6371.0
        
        is_class_5 = student_df['baseline_class'] == 5
        is_class_8 = student_df['baseline_class'] == 8
        
        student_df['is_terminal_transition_risk'] = np.where((is_class_5 & (student_df['dist_up_km'] >= 3.0)) | (is_class_8 & (student_df['dist_sec_km'] >= 3.0)), 1, 0)
        student_df['distance_to_next_school_km'] = np.where(is_class_5, student_df['dist_up_km'], np.where(is_class_8, student_df['dist_sec_km'], np.nan))

    if valid_female_coords.any() and not girls_up_schools.empty and not girls_sec_schools.empty:
        female_coords = np.radians(student_df.loc[valid_female_coords, ['house_latitude', 'house_longitude']].values)
        
        tree_up_g = BallTree(np.radians(girls_up_schools[['latitude', 'longitude']].values), metric='haversine')
        dist_up_g, _ = tree_up_g.query(female_coords, k=1)
        
        tree_sec_g = BallTree(np.radians(girls_sec_schools[['latitude', 'longitude']].values), metric='haversine')
        dist_sec_g, _ = tree_sec_g.query(female_coords, k=1)
        
        student_df.loc[valid_female_coords, 'dist_up_g_km'] = dist_up_g.flatten() * 6371.0
        student_df.loc[valid_female_coords, 'dist_sec_g_km'] = dist_sec_g.flatten() * 6371.0
        
        is_class_5_f = valid_female_coords & (student_df['baseline_class'] == 5)
        is_class_8_f = valid_female_coords & (student_df['baseline_class'] == 8)
        
        student_df['nearest_distance_to_girl_school'] = np.where(is_class_5_f, student_df['dist_up_g_km'], np.where(is_class_8_f, student_df['dist_sec_g_km'], np.nan))
    
    student_df = student_df.drop(columns=['dist_up_km', 'dist_sec_km', 'dist_up_g_km', 'dist_sec_g_km'], errors='ignore')
    return student_df

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'person_id' in df.columns:
        df = df.dropna(subset=['person_id']).copy()

    if 'baseline_class' in df.columns:
        df['baseline_class'] = pd.to_numeric(df['baseline_class'], errors='coerce')
        df = df[df['baseline_class'].between(1, 10)].copy()

    if 'student_district_name' in df.columns and 'student_mandal_name' in df.columns:
        df['is_tribal'] = df.apply(lambda row: 1 if (str(row['student_district_name']).strip().upper(), str(row['student_mandal_name']).strip().upper()) in TRIBAL_MANDALS else 0, axis=1)
    else:
        df['is_tribal'] = 0

    if 'school_management' in df.columns:
        sm = df['school_management'].astype(str).str.upper()
        df['requires_accessible_school'] = sm.str.contains('BLIND|DEAF AND DUMB|MENTALLY RETARDED|ORTHOPAEDICALLY HANDICAPPED|JUVENILLE|BOARSTAL').astype(int)
    else:
        df['requires_accessible_school'] = 0

    anemic_flag = np.where(df.get('is_anemic', pd.Series(['NO']*len(df))).astype(str).str.upper() == 'YES', 1, 0)
    abnormal_flag = np.where(df.get('child_condition', pd.Series(['NORMAL']*len(df))).astype(str).str.upper() == 'ABNORMAL', 1, 0)
    fourd_flag = np.where(df.get('fourd_condition', pd.Series(['--']*len(df))).astype(str).str.title().isin(['Childhood Diseases', 'Defects At Birth', 'Developmental Delays', 'Deficiencies']), 1, 0)
    hb_flag = np.where(df.get('hb_status', pd.Series(['Normal']*len(df))).astype(str).str.title().isin(['Severe', 'Moderate']), 1, 0)
    df['health_vulnerability_score'] = anemic_flag + abnormal_flag + fourd_flag + hb_flag

    uniform = np.where(df.get('uniform_rec', pd.Series(['No']*len(df))).astype(str).str.title() == 'Yes', 1, 0)
    textbooks = np.where(df.get('textbooks_rec', pd.Series(['No']*len(df))).astype(str).str.title() == 'Yes', 1, 0)
    bags = np.where(df.get('bags_rec', pd.Series(['No']*len(df))).astype(str).str.title() == 'Yes', 1, 0)
    shoes = np.where(df.get('shoes_rec', pd.Series(['No']*len(df))).astype(str).str.title() == 'Yes', 1, 0)
    
    df['student_kit_score'] = uniform + textbooks + bags + shoes

    if 'siblings_below_5' in df.columns:
        df['siblings_below_5'] = pd.to_numeric(df['siblings_below_5'], errors='coerce').fillna(0).astype(int)
    else:
        df['siblings_below_5'] = 0

    if 'sanctioned_teachers' in df.columns and 'total_teachers' in df.columns:
        sanc = pd.to_numeric(df['sanctioned_teachers'], errors='coerce').fillna(0)
        tot = pd.to_numeric(df['total_teachers'], errors='coerce').fillna(0)
        df['teacher_deficit_ratio'] = np.where(sanc > 0, (sanc - tot) / sanc, 0)
        df['teacher_deficit_ratio'] = np.clip(df['teacher_deficit_ratio'], 0, 1)
    else:
        df['teacher_deficit_ratio'] = 0.0
        
    for col in ['has_drinking_water', 'has_girls_toilet', 'has_electricity']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
            df[col] = np.where(df[col].isin(['true', 'yes', '1', '1.0']), 1, 0)
        else:
            df[col] = 0

    # --- STRICT DROPOUT (IS_CHURNED) LOGIC ---
    if all(col in df.columns for col in ['baseline_status', 'target_status', 'baseline_class']):
        ssdc = pd.to_numeric(df.get('ssdc_active', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        cg = df.get('course_group_desc', pd.Series('', index=df.index)).astype(str).str.upper()
        ub = df.get('ub_name', pd.Series('', index=df.index)).astype(str).str.upper()
        
        target_str = df['target_status'].astype(str).str.lower()
        is_dropout_target = target_str.isin(['no', 'na', 'none', 'null', 'nan', ''])
        
        is_valid_dropout = is_dropout_target & (ssdc != 1) & (~cg.str.contains('ITI')) & (~ub.str.contains('ITI'))
        
        df['is_churned'] = np.where(
            (df['baseline_status'].notnull()) & 
            (df['baseline_status'].astype(str).str.lower() == 'yes') & 
            is_valid_dropout & 
            (~df['baseline_class'].isin([10, 12, '10', '12'])), 
            1, 0
        )
    elif 'is_churned' not in df.columns:
        df['is_churned'] = 0

    # --- STRICT REJOINER LOGIC (MATCHING EXCEL MATRIX) ---
    h1 = df.get('historical_status_1', pd.Series('unknown', index=df.index)).astype(str).str.lower().str.strip()
    h2 = df.get('historical_status_2', pd.Series('unknown', index=df.index)).astype(str).str.lower().str.strip()
    h3 = df.get('historical_status_3', pd.Series('unknown', index=df.index)).astype(str).str.lower().str.strip()

    if 'baseline_status' in df.columns:
        is_active = (df['baseline_status'].notnull()) & (df['baseline_status'].astype(str).str.lower() == 'yes')
        
        # Check if immediate previous year was No / Missing
        h1_is_no = h1.isin(['no', 'na', 'none', 'null', 'nan', '', 'unknown'])
        # Check if they existed in any older year
        has_older_history = (h2 == 'yes') | (h3 == 'yes')
        
        df['is_recent_rejoiner'] = np.where(is_active & h1_is_no & has_older_history, 1, 0)
    elif 'is_recent_rejoiner' not in df.columns:
        df['is_recent_rejoiner'] = 0

    df['historical_missing_years'] = ((h1 != 'yes') & ((h2 == 'yes') | (h3 == 'yes'))).astype(int) + ((h2 != 'yes') & (h3 == 'yes')).astype(int)

    for yr in ['academic_22', 'academic_23', 'academic_24', 'academic_25']:
        if yr in df.columns: df[f'{yr}_active'] = np.where(df[yr].astype(str).str.lower() == 'yes', 1, 0)

    if 'student_age' in df.columns and 'baseline_class' in df.columns:
        df['student_age'] = pd.to_numeric(df['student_age'], errors='coerce')
        is_biologically_impossible = (df['student_age'] > 25) | (df['student_age'] < 4)
        lower_limit = df['baseline_class'] + 4
        upper_limit = df['baseline_class'] + 6
        df['age_invalid_flag'] = np.where(
            df['baseline_class'].notnull(),
            (df['student_age'] < lower_limit) | (df['student_age'] > upper_limit),
            is_biologically_impossible
        ).astype(int)
        df['student_age'] = np.where(is_biologically_impossible, np.nan, df['student_age'])
    else:
        df['age_invalid_flag'] = 0

    if 'income_band' in df.columns:
        df['income_band'] = df['income_band'].astype(str).str.replace('?', '', regex=False).str.replace('\u2013', '-', regex=False).replace(['nan', 'None', '', 'null'], np.nan)

    if 'marital_status' in df.columns:
        df['marital_status'] = df['marital_status'].astype(str).str.strip().str.lower().map({
            'single': 'Unmarried', 'single woman': 'Unmarried', 'unmarried': 'Unmarried', 'widow': 'Widowed',
            'widower': 'Widowed', 'widowed': 'Widowed', 'married': 'Married', 'divorced': 'Divorced',
            'separated': 'Separated', 'nan': 'Unknown', 'none': 'Unknown', 'null': 'Unknown', '': 'Unknown'
        }).fillna('Unknown')

    mother_present = df['mother_person_id'].notnull() if 'mother_person_id' in df.columns else pd.Series(False, index=df.index)
    father_present = df['father_person_id'].notnull() if 'father_person_id' in df.columns else pd.Series(False, index=df.index)
    df['parental_presence'] = np.select([mother_present & father_present, mother_present & ~father_present, ~mother_present & father_present], ['Both IDs Present', 'Mother ID Only', 'Father ID Only'], default='No IDs Present')

    for col in ['current_attendance', 'previous_attendance', 'current_fa_marks', 'previous_fa_marks', 'current_sa_marks', 'previous_sa_marks', 'members_count', 'house_latitude', 'house_longitude', 'school_latitude', 'school_longitude']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'current_attendance' in df.columns and 'previous_attendance' in df.columns: df['attendance_trend'] = df['current_attendance'] - df['previous_attendance']
    if 'current_fa_marks' in df.columns and 'previous_fa_marks' in df.columns: df['fa_marks_trend'] = df['current_fa_marks'] - df['previous_fa_marks']
    if 'current_sa_marks' in df.columns and 'previous_sa_marks' in df.columns: df['sa_marks_trend'] = df['current_sa_marks'] - df['previous_sa_marks']
        
    if all(c in df.columns for c in ['house_latitude', 'house_longitude', 'school_latitude', 'school_longitude']):
        df['distance_to_school_km'] = np.where(calculate_distance(df['house_latitude'], df['house_longitude'], df['school_latitude'], df['school_longitude']) > 100, np.nan, calculate_distance(df['house_latitude'], df['house_longitude'], df['school_latitude'], df['school_longitude']))

    for col in ['student_gender', 'social_category', 'income_band', 'dwelling_type', 'ration_card_type', 'parental_presence', 'school_category', 'current_thalliki_vandanam_status', 'rural_urban_flag', 'marital_status', 'medium_of_instruction']:
        if col in df.columns: df[col] = df[col].astype(str).replace(['nan', 'None', '', 'null'], 'Unknown').astype('category')

    return df


def apply_manual_weight_matrix(df):
    scores = np.zeros(len(df))
    
    cls = df['baseline_class'].fillna(0)
    scores += np.where(cls.isin([5, 8, 9]), 100, np.where(cls.isin([6, 7]), 40, 0)) * 0.0850
    
    upper_limit = cls + 6
    gap = df['student_age'].fillna(upper_limit) - upper_limit 
    scores += np.where(gap >= 3, 100, np.where(gap == 2, 60, 0)) * 0.0800
    
    att = df['current_attendance'].fillna(100)
    scores += np.where(att < 50, 100, np.where(att < 70, 70, np.where(att < 85, 30, 0))) * 0.0700
    
    fam = df['current_fa_marks'].fillna(100)
    scores += np.where(fam < 35, 100, np.where(fam < 50, 70, np.where(fam < 75, 30, 0))) * 0.0650
    
    sch = df['school_category'].astype(str).str.upper()
    scores += np.where(sch.str.contains('UNKNOWN', na=True), 100, np.where(sch == 'UPPER PRIMARY ONLY', 5, np.where(sch.str.contains('PRIMARY'), 3, 0))) * 0.0550
    
    smt = df['sa_marks_trend'].fillna(0)
    scores += np.where(smt <= -10, 100, np.where(smt <= -9, 40, 0)) * 0.0450
    
    atr = df['attendance_trend'].fillna(0)
    scores += np.where(atr <= -20, 100, np.where(atr <= -10, 40, 0)) * 0.0450
    
    fmt = df['fa_marks_trend'].fillna(0)
    scores += np.where(fmt <= -10, 100, np.where(fmt <= -9, 40, 0)) * 0.0400
    
    dst = df['distance_to_school_km'].fillna(0)
    scores += np.where(dst > 10, 100, np.where(dst >= 6, 75, np.where(dst >= 3, 40, 0))) * 0.0350
    
    tdr = df['teacher_deficit_ratio'].fillna(0)
    scores += np.where(tdr >= 0.50, 100, np.where(tdr >= 0.25, 50, 0)) * 0.0350
    
    med = df['medium_of_instruction'].astype(str).str.lower()
    scores += np.where(med == 'tamil', 100, np.where(med == 'urdu', 40, np.where(med.isin(['english', 'telugu']), 20, 0))) * 0.0350
    
    sam = df['current_sa_marks'].fillna(100)
    scores += np.where(sam < 35, 100, np.where(sam < 50, 70, np.where(sam < 75, 30, 0))) * 0.0300
    
    par = df['parental_presence'].astype(str)
    scores += np.where(par == 'Both IDs Present', 0, np.where(par.isin(['Mother ID Only', 'Father ID Only']), 50, 100)) * 0.0300
    
    scores += np.where(df['current_thalliki_vandanam_status'].astype(str).str.lower() == 'yes', 0, 100) * 0.0300
    
    dwl = df['dwelling_type'].astype(str).str.lower()
    scores += np.where(dwl.str.contains('flat') | (dwl == 'unknown'), 100, np.where(dwl.str.contains('duplex'), 39, np.where(dwl.str.contains('kaccha'), 10, 0))) * 0.0250
    
    mem = df['members_count'].fillna(4)
    scores += np.where((mem == 1) | (mem >= 15), 100, np.where((mem >= 9) & (mem <= 14), 50, 0)) * 0.0150
    
    scores += np.where(df['age_invalid_flag'].fillna(0) == 1, 100, 0) * 0.0150
    
    sib = df['siblings_below_5'].fillna(0)
    scores += np.where(sib >= 2, 100, np.where(sib == 1, 50, 0)) * 0.0200 
    
    jvk = df['student_kit_score'].fillna(4)
    scores += np.where(jvk == 0, 100, np.where(jvk <= 2, 50, 0)) * 0.0200
    
    d_next = df['distance_to_next_school_km'].fillna(0)
    scores += np.where(d_next > 10, 100, np.where(d_next >= 6, 75, np.where(d_next >= 3, 40, 0))) * 0.0200

    soc = df['social_category'].astype(str).str.upper()
    scores += np.where(soc == 'SC', 0, np.where(soc == 'BC', 7, np.where(soc == 'ST', 24, np.where(soc == 'OC', 45, 100)))) * 0.0150
    
    scores += np.where(df['marital_status'].astype(str).str.lower().isin(['unmarried', 'unknown']), 0, 100) * 0.0150
    
    hv = df['health_vulnerability_score'].fillna(0)
    scores += np.where(hv >= 3, 100, np.where(hv == 2, 70, np.where(hv == 1, 30, 0))) * 0.0150
    
    is_fem = df['student_gender'].astype(str).str.lower() == 'female'
    d_girl = df['nearest_distance_to_girl_school'].fillna(0)
    scores += np.where(is_fem & (d_girl > 3), 100, 0) * 0.0100
    
    scores += np.where(df['has_drinking_water'].fillna(1) == 0, 100, 0) * 0.0100
    scores += np.where(df['ration_card_type'].astype(str).str.upper().isin(['AAY', 'BPL']), 0, 100) * 0.0100
    scores += np.where(df['rural_urban_flag'].astype(str).str.upper() == 'R', 0, 100) * 0.0100
    scores += np.where(df['is_tribal'].fillna(0) == 1, 100, 0) * 0.0100
    
    inc = df['income_band'].astype(str).str.replace(' ', '')
    scores += np.where(inc == '>5L', 0, np.where(inc == '3L-5L', 10, np.where(inc == '0-1L', 25, np.where(inc == '1L-3L', 33, 100)))) * 0.0100
    
    scores += np.where(is_fem & (df['has_girls_toilet'].fillna(1) == 0), 100, 0) * 0.0100
    
    gen = df['student_gender'].astype(str).str.lower()
    scores += np.where(gen == 'female', 0, np.where(gen == 'male', 4, np.where(gen == 'transgender', 31, 100))) * 0.0100
    
    scores += np.where(df['is_recent_rejoiner'].fillna(0) == 1, 100, 0) * 0.0100
    
    hmy = df['historical_missing_years'].fillna(0)
    scores += np.where(hmy >= 2, 100, np.where(hmy == 1, 60, 0)) * 0.0100
    
    scores += np.where(df['is_terminal_transition_risk'].fillna(0) == 1, 100, 0) * 0.0100
    
    scores += np.where(df['requires_accessible_school'].fillna(0) == 1, 100, 0) * 0.0500

    df['manual_risk_score'] = np.clip(scores, 0, 100)
    df['stakeholder_risk_level'] = np.where(df['manual_risk_score'] >= 81, 'Critical',
                                   np.where(df['manual_risk_score'] >= 61, 'High',
                                   np.where(df['manual_risk_score'] >= 31, 'Medium', 'Low')))
    return df