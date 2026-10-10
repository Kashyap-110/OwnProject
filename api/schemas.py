from pydantic import BaseModel
from typing import Optional

class StudentPredictionRequest(BaseModel):
    # Base Demographics
    student_age: Optional[float] = None
    student_gender: Optional[str] = 'Unknown'
    social_category: Optional[str] = 'Unknown'
    marital_status: Optional[str] = 'Unknown'
    
    # Household / Economic
    income_band: Optional[str] = 'Unknown'
    dwelling_type: Optional[str] = 'Unknown'
    ration_card_type: Optional[str] = 'Unknown'
    members_count: Optional[float] = 0.0
    siblings_below_5: Optional[int] = 0
    current_thalliki_vandanam_status: Optional[str] = 'Unknown'
    rural_urban_flag: Optional[str] = 'Unknown'
    
    # Academic & Infrastructure
    school_category: Optional[str] = 'Unknown'
    school_management: Optional[str] = 'Unknown'
    school_type_coed_boys_girls: Optional[str] = 'Unknown'
    medium_of_instruction: Optional[str] = 'Unknown'
    baseline_class: Optional[float] = None
    
    # New Infrastructure Features
    sanctioned_teachers: Optional[float] = 0.0
    total_teachers: Optional[float] = 0.0
    has_drinking_water: Optional[int] = 1
    has_girls_toilet: Optional[int] = 1
    has_electricity: Optional[int] = 1
    
    # Welfare (JVK Kit) Features
    uniform_rec: Optional[str] = 'No'
    textbooks_rec: Optional[str] = 'No'
    bags_rec: Optional[str] = 'No'
    shoes_rec: Optional[str] = 'No'
    
    # Location Identifiers for Tribal Logic
    student_district_name: Optional[str] = 'Unknown'
    student_mandal_name: Optional[str] = 'Unknown'
    
    # Health Indicators
    is_anemic: Optional[str] = 'Unknown'
    child_condition: Optional[str] = 'Unknown'
    fourd_condition: Optional[str] = 'Unknown'
    hb_status: Optional[str] = 'Unknown'
    
    # Raw Performance & Attendance Metrics
    current_attendance: Optional[float] = None
    previous_attendance: Optional[float] = None
    current_fa_marks: Optional[float] = None
    previous_fa_marks: Optional[float] = None
    current_sa_marks: Optional[float] = None
    previous_sa_marks: Optional[float] = None
    
    # Raw Location Metrics
    house_latitude: Optional[float] = None
    house_longitude: Optional[float] = None
    school_latitude: Optional[float] = None
    school_longitude: Optional[float] = None
    
    # Pre-Calculated Features
    attendance_trend: Optional[float] = None
    fa_marks_trend: Optional[float] = None
    sa_marks_trend: Optional[float] = None
    distance_to_school_km: Optional[float] = None
    distance_to_next_school_km: Optional[float] = None 
    nearest_distance_to_girl_school: Optional[float] = None 
    
    # Engineered Features
    parental_presence: Optional[str] = 'Unknown'
    historical_missing_years: Optional[int] = 0
    is_recent_rejoiner: Optional[int] = 0
    is_terminal_transition_risk: Optional[int] = 0
    requires_accessible_school: Optional[int] = 0  # <--- Added Consolidated S.N. Flag
    
    # Historical Lookback
    baseline_status: Optional[str] = 'yes'
    historical_status_1: Optional[str] = None  
    historical_status_2: Optional[str] = None  
    historical_status_3: Optional[str] = None  
    
    # Internal Flags 
    age_invalid_flag: Optional[int] = 0