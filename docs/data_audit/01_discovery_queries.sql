-- =============================================================================
-- Student Dropout System v2 - Data Audit Queries (READ-ONLY)
-- Table : aiml_usecase_curated.final_student_dropout_info_new
--
-- Run each query separately (Hue / impala-shell) and paste the output back.
-- Text, CSV or screenshots are all fine.
--
-- Priority 1 (needed before the design is fixed): Q1 - Q9
-- Priority 2 (needed for feature design)        : Q10 - Q13
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Q1. GRAIN: rows vs distinct students per academic year
-- Decides: is the table really one row per student per year?
-- -----------------------------------------------------------------------------
SELECT academic_year,
       COUNT(*)                  AS n_rows,
       COUNT(DISTINCT person_id) AS n_distinct_persons,
       SUM(CASE WHEN person_id IS NULL OR TRIM(person_id) = '' THEN 1 ELSE 0 END) AS n_null_person_id
FROM aiml_usecase_curated.final_student_dropout_info_new
GROUP BY academic_year
ORDER BY academic_year;


-- -----------------------------------------------------------------------------
-- Q1b. DUPLICATES: students with more than one row in the same year
-- Decides: de-duplication rule (mid-year transfer vs. true duplicate).
-- -----------------------------------------------------------------------------
SELECT academic_year,
       rows_per_person,
       COUNT(*)         AS n_persons,
       SUM(diff_class)  AS n_with_different_classes,
       SUM(diff_school) AS n_with_different_schools
FROM (
    SELECT academic_year,
           person_id,
           COUNT(*) AS rows_per_person,
           CASE WHEN MIN(class_currently_enrolled) <> MAX(class_currently_enrolled) THEN 1 ELSE 0 END AS diff_class,
           CASE WHEN MIN(school_id) <> MAX(school_id) THEN 1 ELSE 0 END AS diff_school
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE person_id IS NOT NULL
    GROUP BY academic_year, person_id
) x
GROUP BY academic_year, rows_per_person
ORDER BY academic_year, rows_per_person;


-- -----------------------------------------------------------------------------
-- Q2. ENROLMENT BY CLASS x YEAR
-- Decides: is the 2026-27 roster complete? which class codes exist (-2..13)?
-- -----------------------------------------------------------------------------
SELECT class_currently_enrolled AS cls,
       SUM(CASE WHEN academic_year = '2022-23' THEN 1 ELSE 0 END) AS y2022_23,
       SUM(CASE WHEN academic_year = '2023-24' THEN 1 ELSE 0 END) AS y2023_24,
       SUM(CASE WHEN academic_year = '2024-25' THEN 1 ELSE 0 END) AS y2024_25,
       SUM(CASE WHEN academic_year = '2025-26' THEN 1 ELSE 0 END) AS y2025_26,
       SUM(CASE WHEN academic_year = '2026-27' THEN 1 ELSE 0 END) AS y2026_27,
       SUM(CASE WHEN academic_year IS NULL
                  OR academic_year NOT IN ('2022-23','2023-24','2024-25','2025-26','2026-27')
                THEN 1 ELSE 0 END)                                AS other_or_null_year
FROM aiml_usecase_curated.final_student_dropout_info_new
WHERE person_id IS NOT NULL
GROUP BY class_currently_enrolled
ORDER BY cls;


-- -----------------------------------------------------------------------------
-- Q3. YEAR-TO-YEAR TRANSITION MATRIX  (the core of the dropout label)
-- For each base year T and class: how many are found in T+1, in which class,
-- and how many are missing in T+1 but come back in T+2 (temporary dropouts).
-- Decides: dropout rates per class, roster completeness of 2026-27 (base 2025),
--          promotion / repetition patterns, data errors (lower class, skips).
-- -----------------------------------------------------------------------------
WITH py AS (
    SELECT person_id,
           CAST(SUBSTR(academic_year, 1, 4) AS INT) AS yr,
           MAX(class_currently_enrolled)            AS cls
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE person_id IS NOT NULL
    GROUP BY person_id, CAST(SUBSTR(academic_year, 1, 4) AS INT)
)
SELECT a.yr  AS base_year,
       a.cls AS base_class,
       COUNT(*) AS enrolled,
       SUM(CASE WHEN n1.person_id IS NULL THEN 1 ELSE 0 END)                         AS not_found_next_year,
       ROUND(100 * AVG(CASE WHEN n1.person_id IS NULL THEN 1 ELSE 0 END), 2)         AS pct_not_found,
       SUM(CASE WHEN n1.cls = a.cls + 1 THEN 1 ELSE 0 END)                           AS promoted,
       SUM(CASE WHEN n1.cls = a.cls THEN 1 ELSE 0 END)                               AS same_class,
       SUM(CASE WHEN n1.cls < a.cls THEN 1 ELSE 0 END)                               AS lower_class,
       SUM(CASE WHEN n1.cls > a.cls + 1 THEN 1 ELSE 0 END)                           AS skipped_ahead,
       SUM(CASE WHEN n1.person_id IS NULL AND n2.person_id IS NOT NULL THEN 1 ELSE 0 END) AS missing_next_but_back_in_2_yrs
FROM py a
LEFT JOIN py n1 ON n1.person_id = a.person_id AND n1.yr = a.yr + 1
LEFT JOIN py n2 ON n2.person_id = a.person_id AND n2.yr = a.yr + 2
WHERE a.cls BETWEEN 1 AND 10
  AND a.yr BETWEEN 2022 AND 2025
GROUP BY a.yr, a.cls
ORDER BY a.yr, a.cls;


-- -----------------------------------------------------------------------------
-- Q3b. NOT-FOUND RATE BY SCHOOL MANAGEMENT (base 2024-25, classes 1-9)
-- Decides: do private-school students "vanish" more (coverage gap / migration)?
-- -----------------------------------------------------------------------------
WITH py AS (
    SELECT person_id,
           CAST(SUBSTR(academic_year, 1, 4) AS INT) AS yr
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE person_id IS NOT NULL
    GROUP BY person_id, CAST(SUBSTR(academic_year, 1, 4) AS INT)
)
SELECT b.school_management,
       COUNT(*)                                                               AS enrolled_2024_25,
       SUM(CASE WHEN n.person_id IS NULL THEN 1 ELSE 0 END)                   AS not_found_2025_26,
       ROUND(100 * AVG(CASE WHEN n.person_id IS NULL THEN 1 ELSE 0 END), 2)   AS pct_not_found
FROM aiml_usecase_curated.final_student_dropout_info_new b
LEFT JOIN py n ON n.person_id = b.person_id AND n.yr = 2025
WHERE b.academic_year = '2024-25'
  AND b.class_currently_enrolled BETWEEN 1 AND 9
  AND b.person_id IS NOT NULL
GROUP BY b.school_management
ORDER BY enrolled_2024_25 DESC;


-- -----------------------------------------------------------------------------
-- Q4. WHERE DO CLASS 10 STUDENTS GO NEXT YEAR?
-- Decides: can class 10 be in the dropout target, or is "not found" mostly
--          ITI / polytechnic / supplementary-exam students outside this table?
-- -----------------------------------------------------------------------------
WITH py AS (
    SELECT person_id,
           CAST(SUBSTR(academic_year, 1, 4) AS INT) AS yr,
           MAX(class_currently_enrolled)            AS cls
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE person_id IS NOT NULL
    GROUP BY person_id, CAST(SUBSTR(academic_year, 1, 4) AS INT)
)
SELECT a.yr AS base_year,
       CASE WHEN n.person_id IS NULL THEN 'NOT FOUND' ELSE CAST(n.cls AS STRING) END AS next_year_class,
       COUNT(*) AS n_students
FROM py a
LEFT JOIN py n ON n.person_id = a.person_id AND n.yr = a.yr + 1
WHERE a.cls = 10
  AND a.yr BETWEEN 2022 AND 2025
GROUP BY 1, 2
ORDER BY 1, 2;


-- -----------------------------------------------------------------------------
-- Q5. MONTHLY ATTENDANCE AVAILABILITY (% of rows with working_days > 0)
-- Decides: academic-year calendar (Jun..May), how far 2026-27 data has
--          arrived, and whether future months are 0 or NULL.
-- -----------------------------------------------------------------------------
SELECT academic_year,
       COUNT(*) AS n_rows,
       ROUND(100 * AVG(CASE WHEN jun_working_days > 0 THEN 1 ELSE 0 END), 1) AS jun,
       ROUND(100 * AVG(CASE WHEN jul_working_days > 0 THEN 1 ELSE 0 END), 1) AS jul,
       ROUND(100 * AVG(CASE WHEN aug_working_days > 0 THEN 1 ELSE 0 END), 1) AS aug,
       ROUND(100 * AVG(CASE WHEN sep_working_days > 0 THEN 1 ELSE 0 END), 1) AS sep,
       ROUND(100 * AVG(CASE WHEN oct_working_days > 0 THEN 1 ELSE 0 END), 1) AS oct,
       ROUND(100 * AVG(CASE WHEN nov_working_days > 0 THEN 1 ELSE 0 END), 1) AS nov,
       ROUND(100 * AVG(CASE WHEN dec_working_days > 0 THEN 1 ELSE 0 END), 1) AS dec_,
       ROUND(100 * AVG(CASE WHEN jan_working_days > 0 THEN 1 ELSE 0 END), 1) AS jan,
       ROUND(100 * AVG(CASE WHEN feb_working_days > 0 THEN 1 ELSE 0 END), 1) AS feb,
       ROUND(100 * AVG(CASE WHEN mar_working_days > 0 THEN 1 ELSE 0 END), 1) AS mar,
       ROUND(100 * AVG(CASE WHEN apr_working_days > 0 THEN 1 ELSE 0 END), 1) AS apr,
       ROUND(100 * AVG(CASE WHEN may_working_days > 0 THEN 1 ELSE 0 END), 1) AS may_
FROM aiml_usecase_curated.final_student_dropout_info_new
WHERE class_currently_enrolled BETWEEN 1 AND 10
GROUP BY academic_year
ORDER BY academic_year;


-- -----------------------------------------------------------------------------
-- Q5b. ATTENDANCE DATA QUALITY
-- Decides: cleaning rules (attended > working days, NULL attended, empty rows).
-- -----------------------------------------------------------------------------
SELECT academic_year,
       COUNT(*) AS n_rows,
       SUM(CASE WHEN jun_attended > jun_working_days OR jul_attended > jul_working_days
                  OR aug_attended > aug_working_days OR sep_attended > sep_working_days
                  OR oct_attended > oct_working_days OR nov_attended > nov_working_days
                  OR dec_attended > dec_working_days OR jan_attended > jan_working_days
                  OR feb_attended > feb_working_days OR mar_attended > mar_working_days
                  OR apr_attended > apr_working_days OR may_attended > may_working_days
                THEN 1 ELSE 0 END) AS rows_attended_gt_working_days,
       SUM(CASE WHEN (jun_attended IS NULL AND jun_working_days > 0) OR (jul_attended IS NULL AND jul_working_days > 0)
                  OR (aug_attended IS NULL AND aug_working_days > 0) OR (sep_attended IS NULL AND sep_working_days > 0)
                  OR (oct_attended IS NULL AND oct_working_days > 0) OR (nov_attended IS NULL AND nov_working_days > 0)
                  OR (dec_attended IS NULL AND dec_working_days > 0) OR (jan_attended IS NULL AND jan_working_days > 0)
                  OR (feb_attended IS NULL AND feb_working_days > 0) OR (mar_attended IS NULL AND mar_working_days > 0)
                  OR (apr_attended IS NULL AND apr_working_days > 0) OR (may_attended IS NULL AND may_working_days > 0)
                THEN 1 ELSE 0 END) AS rows_null_attended_with_working_days,
       SUM(CASE WHEN jun_working_days IS NULL AND jul_working_days IS NULL AND aug_working_days IS NULL
                 AND sep_working_days IS NULL AND oct_working_days IS NULL AND nov_working_days IS NULL
                 AND dec_working_days IS NULL AND jan_working_days IS NULL AND feb_working_days IS NULL
                 AND mar_working_days IS NULL AND apr_working_days IS NULL AND may_working_days IS NULL
                THEN 1 ELSE 0 END) AS rows_all_months_null
FROM aiml_usecase_curated.final_student_dropout_info_new
WHERE class_currently_enrolled BETWEEN 1 AND 10
GROUP BY academic_year
ORDER BY academic_year;


-- -----------------------------------------------------------------------------
-- Q6. ATTENDANCE / MARKS COVERAGE BY SCHOOL MANAGEMENT (2025-26, classes 1-10)
-- Decides: can "evidence of attendance" define enrolment for every school type,
--          or do some managements (e.g. private) not report attendance at all?
-- -----------------------------------------------------------------------------
WITH r AS (
    SELECT school_management,
           COALESCE(jun_working_days,0) + COALESCE(jul_working_days,0) + COALESCE(aug_working_days,0)
         + COALESCE(sep_working_days,0) + COALESCE(oct_working_days,0) + COALESCE(nov_working_days,0)
         + COALESCE(dec_working_days,0) + COALESCE(jan_working_days,0) + COALESCE(feb_working_days,0)
         + COALESCE(mar_working_days,0) + COALESCE(apr_working_days,0) + COALESCE(may_working_days,0) AS wd,
           COALESCE(jun_attended,0) + COALESCE(jul_attended,0) + COALESCE(aug_attended,0)
         + COALESCE(sep_attended,0) + COALESCE(oct_attended,0) + COALESCE(nov_attended,0)
         + COALESCE(dec_attended,0) + COALESCE(jan_attended,0) + COALESCE(feb_attended,0)
         + COALESCE(mar_attended,0) + COALESCE(apr_attended,0) + COALESCE(may_attended,0) AS att,
           fa_marks,
           sa_marks
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE academic_year = '2025-26'
      AND class_currently_enrolled BETWEEN 1 AND 10
)
SELECT school_management,
       COUNT(*) AS n_rows,
       ROUND(100 * AVG(CASE WHEN wd > 0 THEN 1 ELSE 0 END), 1)       AS pct_has_working_days,
       ROUND(100 * AVG(CASE WHEN att > 0 THEN 1 ELSE 0 END), 1)      AS pct_attended_any_day,
       ROUND(100 * AVG(CASE WHEN fa_marks > 0 THEN 1 ELSE 0 END), 1) AS pct_fa_gt_0,
       ROUND(100 * AVG(CASE WHEN sa_marks > 0 THEN 1 ELSE 0 END), 1) AS pct_sa_gt_0,
       ROUND(100 * AVG(CASE WHEN att = 0 AND COALESCE(fa_marks,0) = 0 AND COALESCE(sa_marks,0) = 0
                            THEN 1 ELSE 0 END), 1)                   AS pct_no_evidence_at_all
FROM r
GROUP BY school_management
ORDER BY n_rows DESC;


-- -----------------------------------------------------------------------------
-- Q7. "GHOST" ENROLMENTS: rows with no attendance and no marks, and what
--     happens to those students the next year (base years 2022-2025).
-- Decides: does "enrolled" mean "row exists" or "row exists + evidence"?
-- -----------------------------------------------------------------------------
WITH r AS (
    SELECT person_id,
           CAST(SUBSTR(academic_year, 1, 4) AS INT) AS yr,
           class_currently_enrolled                 AS cls,
           CASE WHEN COALESCE(jun_attended,0) + COALESCE(jul_attended,0) + COALESCE(aug_attended,0)
                   + COALESCE(sep_attended,0) + COALESCE(oct_attended,0) + COALESCE(nov_attended,0)
                   + COALESCE(dec_attended,0) + COALESCE(jan_attended,0) + COALESCE(feb_attended,0)
                   + COALESCE(mar_attended,0) + COALESCE(apr_attended,0) + COALESCE(may_attended,0) > 0
                THEN 1 ELSE 0 END AS has_att,
           CASE WHEN COALESCE(fa_marks,0) > 0 OR COALESCE(sa_marks,0) > 0 THEN 1 ELSE 0 END AS has_marks
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE person_id IS NOT NULL
),
py AS (
    SELECT person_id, yr,
           MAX(cls)       AS cls,
           MAX(has_att)   AS has_att,
           MAX(has_marks) AS has_marks
    FROM r
    GROUP BY person_id, yr
)
SELECT a.yr        AS base_year,
       a.has_att   AS base_has_attendance,
       a.has_marks AS base_has_marks,
       COUNT(*)    AS n_students,
       SUM(CASE WHEN b.person_id IS NULL THEN 1 ELSE 0 END)                                     AS not_found_next_year,
       SUM(CASE WHEN b.person_id IS NOT NULL AND b.has_att = 0 AND b.has_marks = 0 THEN 1 ELSE 0 END) AS found_next_year_but_no_evidence
FROM py a
LEFT JOIN py b ON b.person_id = a.person_id AND b.yr = a.yr + 1
WHERE a.cls BETWEEN 1 AND 10
  AND a.yr BETWEEN 2022 AND 2025
GROUP BY a.yr, a.has_att, a.has_marks
ORDER BY a.yr, a.has_att, a.has_marks;


-- -----------------------------------------------------------------------------
-- Q8. WHICH COLUMNS ARE "YEAR-SPECIFIC" AND WHICH ARE "LATEST SNAPSHOTS"?
-- For students with 2+ rows: how many have a value that CHANGES across years.
-- A column that never changes while class changes is a snapshot copied onto
-- every year -> it can leak the future (e.g. highest_qualification, student_age).
-- -----------------------------------------------------------------------------
WITH p AS (
    SELECT person_id,
           COUNT(*) AS n_rows,
           CASE WHEN MIN(class_currently_enrolled) <> MAX(class_currently_enrolled) THEN 1 ELSE 0 END AS v_class,
           CASE WHEN MIN(highest_qualification)    <> MAX(highest_qualification)    THEN 1 ELSE 0 END AS v_highest_qualification,
           CASE WHEN MIN(student_age)              <> MAX(student_age)              THEN 1 ELSE 0 END AS v_student_age,
           CASE WHEN MIN(education_level)          <> MAX(education_level)          THEN 1 ELSE 0 END AS v_education_level,
           CASE WHEN MIN(school_id)                <> MAX(school_id)                THEN 1 ELSE 0 END AS v_school_id,
           CASE WHEN MIN(school_id) = MAX(school_id)
                 AND MIN(total_students) <> MAX(total_students)                     THEN 1 ELSE 0 END AS v_school_size_same_school,
           CASE WHEN MIN(CAST(thalliki_vandanam AS INT)) <> MAX(CAST(thalliki_vandanam AS INT)) THEN 1 ELSE 0 END AS v_thalliki_vandanam,
           CASE WHEN MIN(CAST(is_dropout AS INT))        <> MAX(CAST(is_dropout AS INT))        THEN 1 ELSE 0 END AS v_is_dropout,
           CASE WHEN MIN(marital_status)           <> MAX(marital_status)           THEN 1 ELSE 0 END AS v_marital_status,
           CASE WHEN MIN(income_band)              <> MAX(income_band)              THEN 1 ELSE 0 END AS v_income_band,
           CASE WHEN MIN(ration_card_type)         <> MAX(ration_card_type)         THEN 1 ELSE 0 END AS v_ration_card_type,
           CASE WHEN MIN(dwelling_type)            <> MAX(dwelling_type)            THEN 1 ELSE 0 END AS v_dwelling_type,
           CASE WHEN MIN(members_count)            <> MAX(members_count)            THEN 1 ELSE 0 END AS v_members_count,
           CASE WHEN MIN(siblings_below_5)         <> MAX(siblings_below_5)         THEN 1 ELSE 0 END AS v_siblings_below_5,
           CASE WHEN MIN(child_status)             <> MAX(child_status)             THEN 1 ELSE 0 END AS v_child_status,
           CASE WHEN MIN(bmi)                      <> MAX(bmi)                      THEN 1 ELSE 0 END AS v_bmi,
           CASE WHEN MIN(is_anemic)                <> MAX(is_anemic)                THEN 1 ELSE 0 END AS v_is_anemic,
           CASE WHEN MIN(uniform_rec)              <> MAX(uniform_rec)              THEN 1 ELSE 0 END AS v_uniform_rec,
           CASE WHEN MIN(house_latitude)           <> MAX(house_latitude)           THEN 1 ELSE 0 END AS v_house_latitude,
           CASE WHEN MIN(medium_name)              <> MAX(medium_name)              THEN 1 ELSE 0 END AS v_medium_name,
           CASE WHEN MIN(is_deceased)              <> MAX(is_deceased)              THEN 1 ELSE 0 END AS v_is_deceased
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE person_id IS NOT NULL
    GROUP BY person_id
)
SELECT COUNT(*)                     AS persons_with_2plus_rows,
       SUM(v_class)                 AS class_varies,
       SUM(v_highest_qualification) AS highest_qualification_varies,
       SUM(v_student_age)           AS student_age_varies,
       SUM(v_education_level)       AS education_level_varies,
       SUM(v_school_id)             AS school_id_varies,
       SUM(v_school_size_same_school) AS school_size_varies_same_school,
       SUM(v_thalliki_vandanam)     AS thalliki_vandanam_varies,
       SUM(v_is_dropout)            AS is_dropout_varies,
       SUM(v_marital_status)        AS marital_status_varies,
       SUM(v_income_band)           AS income_band_varies,
       SUM(v_ration_card_type)      AS ration_card_type_varies,
       SUM(v_dwelling_type)         AS dwelling_type_varies,
       SUM(v_members_count)         AS members_count_varies,
       SUM(v_siblings_below_5)      AS siblings_below_5_varies,
       SUM(v_child_status)          AS child_status_varies,
       SUM(v_bmi)                   AS bmi_varies,
       SUM(v_is_anemic)             AS is_anemic_varies,
       SUM(v_uniform_rec)           AS uniform_rec_varies,
       SUM(v_house_latitude)        AS house_latitude_varies,
       SUM(v_medium_name)           AS medium_name_varies,
       SUM(v_is_deceased)           AS is_deceased_varies
FROM p
WHERE n_rows >= 2;


-- -----------------------------------------------------------------------------
-- Q9. LEAKAGE PROBE: next-year "not found" rate by suspicious columns
--     (base 2024-25, classes 1-9, outcome = found in 2025-26 or not).
-- A value with an extreme rate (e.g. 0% vs 60%) is a red flag that the column
-- is recorded AFTER the outcome and must not be used as a feature.
-- -----------------------------------------------------------------------------
WITH py AS (
    SELECT person_id,
           CAST(SUBSTR(academic_year, 1, 4) AS INT) AS yr
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE person_id IS NOT NULL
    GROUP BY person_id, CAST(SUBSTR(academic_year, 1, 4) AS INT)
),
lab AS (
    SELECT b.class_currently_enrolled AS cls,
           b.highest_qualification,
           b.thalliki_vandanam,
           b.child_status,
           b.marital_status,
           b.is_dropout,
           b.dropout_reason,
           b.is_deceased,
           b.uniform_rec,
           CASE WHEN n.person_id IS NULL THEN 1 ELSE 0 END AS not_found_next
    FROM aiml_usecase_curated.final_student_dropout_info_new b
    LEFT JOIN py n ON n.person_id = b.person_id AND n.yr = 2025
    WHERE b.academic_year = '2024-25'
      AND b.class_currently_enrolled BETWEEN 1 AND 9
      AND b.person_id IS NOT NULL
)
SELECT 'thalliki_vandanam' AS feature, CAST(thalliki_vandanam AS STRING) AS value, COUNT(*) AS n,
       ROUND(100 * AVG(not_found_next), 2) AS pct_not_found_next FROM lab GROUP BY 2
UNION ALL
SELECT 'child_status', child_status, COUNT(*), ROUND(100 * AVG(not_found_next), 2) FROM lab GROUP BY 2
UNION ALL
SELECT 'marital_status', marital_status, COUNT(*), ROUND(100 * AVG(not_found_next), 2) FROM lab GROUP BY 2
UNION ALL
SELECT 'is_dropout', CAST(is_dropout AS STRING), COUNT(*), ROUND(100 * AVG(not_found_next), 2) FROM lab GROUP BY 2
UNION ALL
SELECT 'dropout_reason', dropout_reason, COUNT(*), ROUND(100 * AVG(not_found_next), 2) FROM lab GROUP BY 2
UNION ALL
SELECT 'is_deceased', is_deceased, COUNT(*), ROUND(100 * AVG(not_found_next), 2) FROM lab GROUP BY 2
UNION ALL
SELECT 'uniform_rec', uniform_rec, COUNT(*), ROUND(100 * AVG(not_found_next), 2) FROM lab GROUP BY 2
UNION ALL
SELECT 'highest_qual_minus_class', CAST(CAST(highest_qualification AS INT) - cls AS STRING), COUNT(*),
       ROUND(100 * AVG(not_found_next), 2) FROM lab GROUP BY 2
ORDER BY 1, 3 DESC;


-- =============================================================================
-- PRIORITY 2
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Q10. FA / SA MARKS SCALE BY YEAR x CLASS
-- Decides: normalisation (marks are totals; max depends on class/subjects),
--          whether 0 means "absent / not uploaded".
-- -----------------------------------------------------------------------------
SELECT academic_year,
       class_currently_enrolled AS cls,
       COUNT(*) AS n_rows,
       ROUND(100 * AVG(CASE WHEN fa_marks IS NULL THEN 1 ELSE 0 END), 1) AS fa_null_pct,
       ROUND(100 * AVG(CASE WHEN fa_marks = 0     THEN 1 ELSE 0 END), 1) AS fa_zero_pct,
       ROUND(APPX_MEDIAN(CAST(fa_marks AS DOUBLE)), 1)                    AS fa_median,
       ROUND(MAX(CAST(fa_marks AS DOUBLE)), 1)                            AS fa_max,
       ROUND(100 * AVG(CASE WHEN sa_marks IS NULL THEN 1 ELSE 0 END), 1) AS sa_null_pct,
       ROUND(100 * AVG(CASE WHEN sa_marks = 0     THEN 1 ELSE 0 END), 1) AS sa_zero_pct,
       ROUND(APPX_MEDIAN(sa_marks), 1)                                    AS sa_median,
       ROUND(MAX(sa_marks), 1)                                            AS sa_max
FROM aiml_usecase_curated.final_student_dropout_info_new
WHERE class_currently_enrolled BETWEEN 1 AND 10
GROUP BY academic_year, class_currently_enrolled
ORDER BY academic_year, cls;


-- -----------------------------------------------------------------------------
-- Q11. CATEGORICAL VALUE DISTRIBUTIONS (2025-26, classes 1-10)
-- Decides: cleaning / grouping of categories, garbage values (income_band).
-- -----------------------------------------------------------------------------
WITH s AS (
    SELECT *
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE academic_year = '2025-26'
      AND class_currently_enrolled BETWEEN 1 AND 10
)
SELECT 'income_band' AS col, income_band AS value, COUNT(*) AS n FROM s GROUP BY 2
UNION ALL SELECT 'household_income_category', household_income_category, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'ration_card_type', ration_card_type, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'dwelling_type', dwelling_type, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'social_category', social_category, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'marital_status', marital_status, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'child_status', child_status, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'school_management', school_management, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'school_category', school_category, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'board_affiliation', board_affiliation, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'medium_name', medium_name, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'student_gender', student_gender, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'rural_urban_flag', rural_urban_flag, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'is_deceased', is_deceased, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'is_anemic', is_anemic, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'child_condition', child_condition, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'hb_status', hb_status, COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'thalliki_vandanam', CAST(thalliki_vandanam AS STRING), COUNT(*) FROM s GROUP BY 2
UNION ALL SELECT 'functional_status', functional_status, COUNT(*) FROM s GROUP BY 2
ORDER BY 1, 3 DESC;


-- -----------------------------------------------------------------------------
-- Q12. IDENTITY STABILITY
-- Decides: is person_id a reliable key across years? (ID changes = fake dropouts)
-- -----------------------------------------------------------------------------
SELECT 'student_id_with_multiple_person_ids' AS check_name, COUNT(*) AS n
FROM (
    SELECT student_id
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE student_id IS NOT NULL AND person_id IS NOT NULL
    GROUP BY student_id
    HAVING MIN(person_id) <> MAX(person_id)
) x
UNION ALL
SELECT 'person_id_with_multiple_student_ids', COUNT(*)
FROM (
    SELECT person_id
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE student_id IS NOT NULL AND person_id IS NOT NULL
    GROUP BY person_id
    HAVING MIN(student_id) <> MAX(student_id)
) y
UNION ALL
SELECT 'rows_with_null_person_id_but_student_id', COUNT(*)
FROM aiml_usecase_curated.final_student_dropout_info_new
WHERE person_id IS NULL AND student_id IS NOT NULL;


-- -----------------------------------------------------------------------------
-- Q13. AGE-FOR-CLASS (age on 1 June of the academic year, from DOB) - 2025-26
-- Decides: over-age feature design and invalid-DOB handling.
-- Typical value of (age - class) is about 5.
-- -----------------------------------------------------------------------------
SELECT CASE WHEN gap IS NULL THEN 'DOB missing'
            WHEN gap <= 2    THEN '<=2'
            WHEN gap >= 11   THEN '>=11'
            ELSE CAST(gap AS STRING) END AS age_minus_class,
       COUNT(*) AS n_students
FROM (
    SELECT CAST(FLOOR(DATEDIFF(CAST(CONCAT(SUBSTR(academic_year, 1, 4), '-06-01') AS TIMESTAMP),
                               CAST(student_date_of_birth AS TIMESTAMP)) / 365.25) AS INT)
           - class_currently_enrolled AS gap
    FROM aiml_usecase_curated.final_student_dropout_info_new
    WHERE academic_year = '2025-26'
      AND class_currently_enrolled BETWEEN 1 AND 10
) x
GROUP BY 1
ORDER BY 1;
