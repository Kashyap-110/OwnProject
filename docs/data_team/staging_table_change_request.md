# Change Request: Staging Table for the Student Dropout Model

| Item | Value |
|---|---|
| Table | `aiml_usecase_curated.final_student_dropout_info_new` |
| From | AI/ML team, Student Dropout System |
| To | Data engineering team |
| Date | 2026-10-09 |
| Data examined | All rows with `load_date` = 2026-10-08 |
| Writing standard | ASD-STE100 (Simplified Technical English) |

---

## 1. Purpose

This document gives the changes that the ML team needs in the staging table.

The staging table supplies the data for the student dropout model. The Government of Andhra Pradesh will use the model to find students at risk of dropout. Field staff will then help these students.

Wrong data gives a wrong list of students. A wrong list sends help to the wrong students. It also hides the students who need help.

---

## 2. How to use this document

- Each requirement has an ID (R01 to R46), a priority, a finding, instructions, a reason, and an acceptance check.
- **Priority 1:** The model cannot be correct without this change. Do these changes first.
- **Priority 2:** This change makes the model much better.
- **Priority 3:** This change makes the model better. Do it if the source data exists.
- A **finding** is a result from our data audit. Q1 to Q13 are the audit queries. The ML team can share the query file (`01_discovery_queries.sql`).
- If a rule is not clear, ask the ML team. Do not guess a rule.

---

## 3. Terms

| Term | Meaning in this document |
|---|---|
| Academic year | The school year from June to May. Example: 2025-26 is June 2025 to May 2026. |
| Row | One record for one student in one academic year. |
| Year value | A value that was true in the academic year of the row. |
| Snapshot value | The latest value from a source, copied into all rows of the student. |
| Carried-forward row | A row that the source system copies into a new academic year without action from the school. |
| Evidence of study | Attendance on 1 or more days, or marks in 1 or more assessments. |
| Label | The value that the model learns to predict. For this model: the student does not continue in the next academic year. |
| Data leakage | Information from the future in a row. It makes the model look correct in tests, but the model fails in real use. |
| Placeholder ID | One ID value that a source uses for many different unknown persons. |
| As-of date | The last date that a value covers. |

**Abbreviations:** AAY (Antyodaya Anna Yojana), APOSS (Andhra Pradesh Open School Society), BIE (Board of Intermediate Education), BPL (Below Poverty Line), CBSE (Central Board of Secondary Education), CSE (Commissionerate of School Education), CWSN (Children With Special Needs), FA (Formative Assessment), ICSE (Indian Certificate of Secondary Education), ITI (Industrial Training Institute), NIOS (National Institute of Open Schooling), SA (Summative Assessment), SBTET (State Board of Technical Education and Training), SSC (Secondary School Certificate, class 10 examination), SSDC (Skill development training), TC (Transfer Certificate), UDISE+ (Unified District Information System for Education Plus).

---

## 4. Main findings of the audit

| No. | Finding | Effect on the model |
|---|---|---|
| 1 | In 2024-25, one `person_id` has 127,571 rows. | 127,571 students cannot be connected across years. They look like dropouts. They also get the household data of a different person. |
| 2 | 2023-24 has only 38,129 class 1 rows. Other years have 591,439 to 694,694. | False dropouts for 2022-23. False "new" students in 2024-25. |
| 3 | Of the 2022-23 students not found in 2023-24, 74.9% come back in 2024-25. | Most of these "dropouts" are rows that are not in the table. |
| 4 | The table keeps students after they leave school. Rows with no attendance and no marks increase every year. | A row in a year does not prove that the student studied. The label cannot use "row exists". |
| 5 | `highest_qualification` and `student_age` are snapshot values. | Direct data leakage. |
| 6 | School, household, and health columns are snapshot values. | The model sees today's values in old rows. |
| 7 | FA and SA marks are totals with a different scale each year. Some values are not possible (for example 1,035,200). | Marks cannot be compared across years and classes. |
| 8 | The table uses many values for "no data": NULL, `'null'`, `''`, `'-'`. Some text has spelling errors and a wrong encoding. | The model learns false categories. |
| 9 | `is_dropout` is TRUE for 4.6% of 2024-25 rows. Only 31.14% of these students are not in the table in 2025-26. | We cannot use this column as the label. |
| 10 | 2026-27 attendance stops at July 2026. The load date is 2026-10-08. | The early-warning list is 2 months late. |

---

## 5. General rules

These rules apply to all columns. The requirements in Section 6 give the details.

1. Keep one row for each student for each academic year (R02).
2. Make sure that each value in a row shows the situation in the academic year of that row (Section 6.D).
3. Use the database NULL for "not known". Do not use other values for "not known" (R33).
4. Give a date for each value that changes in time (R43).
5. Do not calculate the dropout label in the table. Give the facts. The ML team calculates the label (R15).
6. Keep all classes and all school types in the table (R06).

---

## 6. Requirements

### 6.A Table structure and identity

#### R01. Make the changes in a new version of the table
**Priority:** 1

**Instructions:**
1. Make a new table, for example `aiml_usecase_curated.final_student_dropout_info_v2`.
2. Do not change the current table until the ML team accepts the new table.
3. Partition the new table by `academic_year`.

**Reason:** The ML team compares the two tables during the acceptance tests. The current dashboards use the current table. If the new table has a problem, the users can go back to the current table.

**Acceptance check:** Both tables are available. The new table has one partition for each academic year.

---

#### R02. Keep one row for each student for each academic year
**Priority:** 1

**Finding (Q1b):** Some students have more than one row in the same academic year.
- 2022-23: 19,663 students. 18,063 of them have different classes in the same year.
- 2023-24: 8,224 students. 5,994 of them have different schools in the same year.
- 2025-26: 304 students.
- 2026-27: 1,751 students.

**Instructions:**
1. Keep one row for each `person_id` and `academic_year`.
2. If a student changed school during the year, use the last school for the main row.
3. Add the column `enrolments_in_year`. It is the number of school enrolments of the student in the year.
4. Add the columns `first_school_id_in_year` and `school_change_date`.
5. If two rows have different classes in the same school, find the correct class. Correct the row.
6. Write the rule that selects the main row in the data dictionary.

**Reason:** Two rows for one student count the student two times. Two different classes for one student give conflicting information. A school change during the year is a risk signal. The new columns keep this information.

**Acceptance check:** Test T1 (Section 9) returns no rows.

---

#### R03. Remove the placeholder `person_id` from 2024-25
**Priority:** 1

**Finding (Q1, Q1b, Q9, Q11):**
- In 2024-25, one `person_id` has 127,571 rows.
- In 2024-25 (classes 1 to 9), 89,361 rows show `marital_status` = "Widowed". 92,918 rows show `is_deceased` = "Deceased".
- In 2025-26 (classes 1 to 10), these numbers are only 1,430 and 5,760.
- These rows probably get the household data of one adult person through the placeholder ID.

**Instructions:**
1. Find the `person_id` value that has 127,571 rows in 2024-25.
2. For each of these rows, find the correct `person_id`. Use `student_id` and the source records.
3. If you cannot find the correct `person_id`, set `person_id` to NULL.
4. For these rows, set the new column `person_id_status` to `UNMATCHED`.
5. Do not use one shared value for unknown persons in any year.
6. Find other placeholder values in all years. Examples: `'0'`, `'000000000000'`, `'-1'`.

**Reason:** The model connects the years of a student through `person_id`. Students with a shared ID cannot be connected. They look like dropouts. They also get household data from a different person. This gives false dropouts and false household risk.

**Acceptance check:** Test T2 (Section 9). No `person_id` has more rows than the number of academic years in the table.

---

#### R04. Give one permanent key to each student
**Priority:** 1

**Finding (Q12):**
- 68,088 `student_id` values have more than one `person_id`.
- 201,034 `person_id` values have more than one `student_id`.

**Instructions:**
1. Examine these cases with the source system teams.
2. Give each child one permanent key for all academic years. Use `person_id` as this key.
3. Merge the rows of one child that have different `person_id` values.
4. Keep the `student_id` of each academic year in the row.
5. Give a mapping table with these columns: `old_id`, `id_type`, `person_id`, `change_date`, `change_reason`.

**Reason:** If a child gets a new ID, the old ID looks like a dropout. The new ID looks like a new student. This gives false dropouts and false new students.

**Acceptance check:** Test T3 (Section 9) shows the number of conflicts before and after the change. The data team writes the reason for each type of conflict that stays.

---

#### R05. Keep ID columns as text
**Priority:** 1

**Finding:** `person_id` has leading zeros (example: `096904697373`). An Excel extract shows the same ID as `96904697373`.

**Instructions:**
1. Keep `person_id`, `student_id`, `school_id`, and `household_id` as STRING.
2. Do not change these IDs to a number type in any step.
3. Keep the leading zeros.

**Reason:** A number type removes the leading zeros. Then a join can fail, or two different IDs can become equal.

**Acceptance check:** The data type of each ID column is STRING. The IDs with leading zeros are unchanged.

---

#### R06. Keep all classes and all school types
**Priority:** 1

**Instructions:**
1. Keep the rows for pre-primary classes and for classes 1 to 12.
2. Keep the rows of all school managements, including private schools.
3. Do not filter the table to classes 1 to 10.

**Reason:** The model predicts dropout for classes 1 to 10. But the model needs the row of the next year to know the result. A class 10 student continues in class 11. Pre-primary rows show where class 1 students come from.

**Acceptance check:** Test Q2 shows rows for all classes in all years.

---

#### R07. Write the meaning of each class code
**Priority:** 1

**Finding (Q2):**
- `class_currently_enrolled` has the values -3, -2, -1, 0, 1 to 12, and 13.
- Class 13 has 449,134 rows in 2024-25, 888,888 rows in 2025-26, and 1,381,411 rows in 2026-27.
- One sample row with class 13 shows age 63 and the reason "Class 12 - Passed Out".
- Class 0 has 1 row (2023-24).

**Instructions:**
1. Write the meaning of each class code in the data dictionary.
2. If class 13 means "completed class 12", do not keep these persons as enrolled students.
3. Put the completion status in the status columns (R12).
4. Correct the row with class 0.

**Reason:** The model must know which rows are real school enrolments. The number of class 13 rows increases each year. This agrees with old students who stay in the table after they complete school.

**Acceptance check:** Each class code has a meaning in the data dictionary. Class 13 rows are not enrolment rows.

---

### 6.B Completeness and timeliness

#### R08. Load the missing 2023-24 enrolment rows
**Priority:** 1

**Finding (Q2, Q3):**
- 2023-24 has 38,129 class 1 rows. Other years have 591,439 to 694,694 class 1 rows.
- 2023-24 has 650 class -2 rows and 4,524 class -1 rows. Other years have 118,611 to 314,054 rows for each of these classes.
- 122,488 students of 2022-23 (classes 1 to 10) are not in 2023-24. 91,688 of them (74.9%) come back in 2024-25.
- For class 1 of 2022-23, 94.0% of the students not found in 2023-24 come back in 2024-25.
- For comparison: of the 2023-24 students not found in 2024-25, only 12.4% come back in 2025-26.

**Instructions:**
1. Find why 2023-24 has few class 1 and pre-primary rows.
2. Load all 2023-24 enrolment rows from the source.
3. Find the students who have rows in 2022-23 and 2024-25 but not in 2023-24.
4. Load the 2023-24 rows of these students.
5. If the source does not have these rows, tell the ML team. Write the cause in the data dictionary.

**Reason:** A missing row looks like a dropout. The 2022-23 data then has false dropouts. The model learns from these false dropouts. Also, the class 1 students of 2023-24 look like new students in 2024-25.

**Acceptance check:**
- The class 1 count and the pre-primary counts of 2023-24 are similar to the other years.
- Of the 2022-23 students not found in 2023-24, less than 15% come back in 2024-25.

---

#### R09. Load the monthly attendance on time
**Priority:** 1

**Finding (Q5):**
- The load date is 2026-10-08.
- For 2026-27, June has attendance for 84.9% of rows and July for 83.9%.
- August and September 2026 have no attendance.

**Instructions:**
1. Load the attendance of each month not later than 10 days after the end of the month.
2. Add the column `attendance_as_of_date`. It is the last date that the attendance data covers.
3. If 10 days is not possible, agree a different time with the ML team.

**Reason:** The early-warning list uses the latest attendance. A delay of 2 months makes the warnings late. The field staff then act too late.

**Acceptance check:** On the 15th day of each month, the previous month has attendance for most rows.

---

#### R10. Load the June and July 2022 attendance
**Priority:** 3

**Finding (Q5):** In 2022-23, 0.0% of rows have June attendance. 1.4% of rows have July attendance.

**Instructions:**
1. If the source has June and July 2022 attendance, load it.
2. If the source does not have it, write this in the data dictionary.

**Reason:** Without June and July, the first quarter of 2022-23 has almost no attendance. The model then sees a false difference between 2022-23 and the other years.

**Acceptance check:** June and July 2022 have attendance, or the data dictionary tells why not.

---

#### R11. Add older academic years if the source has them
**Priority:** 3

**Instructions:**
1. If the source has 2021-22 data, add it to the table.
2. Use the same columns and the same rules as for the other years.

**Reason:** The model uses the previous year of each student. Now the 2022-23 rows have no previous year. 2021-22 data gives them a previous year. The ML team will use 2021-22 only as history, because the COVID-19 school closures made that year different.

**Acceptance check:** 2021-22 rows exist, or the data team tells the ML team that the source does not have them.

---

### 6.C Student status and outcome facts

**Finding for this section (Q3, Q5b, Q7, Q9):** The source system keeps students in the table after they leave school.
- Students not found in the next year: 1.18% to 4.32% for 2022-23 and 2023-24. But only 0.09% to 0.64% for 2024-25 and 2025-26.
- In 2024-25, 361,166 students (classes 1 to 10) had no attendance and no marks. 96.0% of them have a 2025-26 row, again with no attendance and no marks.
- Each year, about 10% of students with attendance have a next-year row with no attendance and no marks. For 2024-25 to 2025-26, this is 626,219 students.
- In 2024-25, 283,683 students (classes 1 to 9) had `child_status` = "Left School". 93.46% of them have a 2025-26 row.
- Rows with no attendance in any month: 232,455 in 2022-23 and 590,172 in 2025-26.

**Result:** A row in a year does not prove that the student studied in that year. The requirements R12 to R18 give the facts that the label needs.

#### R12. Add the official status of the student for each academic year
**Priority:** 1

**Instructions:**
1. Add the column `student_status_at_year_end`.
2. Use only these values:
   - `PROMOTED`
   - `DETAINED`
   - `LEFT_SCHOOL`
   - `TRANSFERRED`
   - `MIGRATED_OUT_OF_STATE`
   - `DECEASED`
   - `COMPLETED`
   - `NOT_KNOWN`
3. Add the column `student_status_date`. It is the date of the status change.
4. Add the column `leaving_reason` for the status `LEFT_SCHOOL`. Use a fixed code list.
5. Use the school records (CSE Child Info and UDISE+) as the source.
6. Use the same rules for all academic years.

**Reason:** The model must separate a dropout from a transfer, a migration, a death, and a completion. Only a dropout is the target. If these cases are mixed, the labels are wrong.

**Acceptance check:** Test T8 (Section 9). `NOT_KNOWN` is less than 5% of the rows in each completed academic year.

---

#### R13. Mark the rows that the system copied into a new year
**Priority:** 1

**Instructions:**
1. Add the column `record_source`. Use only these values:
   - `NEW_ADMISSION`
   - `PROMOTED_BY_SCHOOL`
   - `CARRIED_FORWARD_BY_SYSTEM`
   - `READMISSION`
   - `TRANSFER_IN`
2. Add the columns `record_created_date` and `record_last_updated_date`.
3. Add the column `school_confirmed_flag`. Set it to TRUE when the school confirmed the student for the year.
4. Write the date in each year when the system copies the students into the new year.

**Reason:** A carried-forward row does not show that the student studied. The ML team must know which rows the school confirmed.

**Acceptance check:** Each row has a `record_source` value. The data dictionary gives the copy date of each year.

---

#### R14. Give the transfer certificate and admission data
**Priority:** 2

**Instructions:**
1. Add the columns `tc_issued_flag` and `tc_issue_date`.
2. Add the column `tc_to_school_id` if the source has it.
3. Add the column `admission_date`. It is the date of admission to the school of the row.

**Reason:** A student with a TC usually goes to a different school. If the new school is not in the table, the student looks like a dropout. The TC data tells the ML team that the student moved.

**Acceptance check:** The TC columns have values for the students with TCs in the source.

---

#### R15. Write the rule for `is_dropout` and `dropout_reason`
**Priority:** 1

**Finding (Q9):**
- In 2024-25 (classes 1 to 9), `is_dropout` is TRUE for 292,542 rows (4.6%).
- Only 31.14% of these students are not in the table in 2025-26.
- `dropout_reason` is NULL for all 2024-25 rows.
- Sample rows show reasons such as "Class 12 - Passed Out" and "Active for Import/Status Not Known".

**Instructions:**
1. Write the rule that sets `is_dropout` and `dropout_reason`.
2. Write the date when the source sets these values.
3. Use the same rule for all academic years.
4. If you cannot make the values correct, rename the columns to `source_is_dropout` and `source_dropout_reason`.
5. Do not add a new calculated dropout label to the table.

**Reason:** The ML team does not use these columns as model inputs. The ML team uses them only to compare with its own label. For this comparison, the ML team must know their meaning. The label is calculated in the model code. One rule in one place prevents two different dropout numbers.

**Acceptance check:** The data dictionary gives the rule and the date for both columns.

---

#### R16. Give the class 10 examination result
**Priority:** 1

**Finding (Q4):**
- Of the 2025-26 class 10 students, 643,875 are in class 11 in 2026-27.
- 176,473 (21.5%) are in class 10 again in 2026-27.
- For 2022-23, this value was only 5.5%.

**Instructions:**
1. Add the column `ssc_result`. Use only these values: `PASS`, `FAIL`, `ABSENT`, `NOT_REGISTERED`.
2. Add the column `ssc_result_date`.
3. Add the columns `supplementary_result` and `supplementary_result_date`.
4. Add the column `ssc_attempt_number`.

**Reason:** Failure in the class 10 examination is a main cause of dropout after class 10. Without the result, the ML team cannot know if a student in class 10 again is a repeater or a carried-forward row.

**Acceptance check:** Each class 10 row of a completed year has an `ssc_result` value.

---

#### R17. Add rows for ITI, polytechnic, open school, and skill training
**Priority:** 1

**Finding:** The table has class 11 and class 12 rows from junior colleges. It does not have ITI, polytechnic, open school, or skill training enrolments. The old staging table had ITI and SSDC columns.

**Instructions:**
1. Add one row for each student in each of these institutions, for each academic year.
2. Add the column `institution_type`. Use these values: `SCHOOL`, `JUNIOR_COLLEGE`, `ITI`, `POLYTECHNIC`, `OPEN_SCHOOL`, `SKILL_TRAINING`.
3. Add the columns `institution_id` and `admission_date` for these rows.
4. Use the sources BIE, ITI admissions, SBTET, APOSS, NIOS, and SSDC, if they are available.
5. Tell the ML team which sources are not available.

**Reason:** A student who joins ITI or a polytechnic is not a dropout. Without these sources, these students look like dropouts. The model then learns wrong labels for classes 9 and 10.

**Acceptance check:** The table has rows with each available `institution_type`. The data dictionary lists the sources that are not available.

---

#### R18. Give attendance for class 11 and class 12
**Priority:** 2

**Instructions:**
1. Tell the ML team if class 11 and class 12 rows have attendance.
2. If they do not have attendance, add attendance from the junior college source.
3. If attendance is not possible, add the column `admission_confirmed_flag`.

**Reason:** The ML team uses class 11 rows to know that a class 10 student continued. A row without attendance does not prove this (see the finding at the start of Section 6.C).

**Acceptance check:** Class 11 and class 12 rows have attendance or `admission_confirmed_flag`.

---

### 6.D Point-in-time correctness (prevention of data leakage)

**Rule for this section:** Each row for an academic year must show the values of that academic year. A value from a later date gives the model information from the future. The model then looks correct in tests, but it fails in real use.

**Finding for this section (Q8):** We examined 10,151,945 students with 2 or more rows.

| Column | Number of students with a value that changes between years | Result |
|---|---|---|
| `class_currently_enrolled` | 9,500,344 | Year value. Correct. |
| `school_id` | 5,146,251 | Year value. Correct. |
| `thalliki_vandanam` | 7,769,128 | Year value. Correct. |
| `medium_name` | 1,378,005 | Year value. Correct. |
| `highest_qualification` | 107,257 (1%) | Snapshot value. |
| `student_age` | 0 | Snapshot value. |
| `total_students` (same school) | 0 | Snapshot value. |
| `marital_status`, `income_band`, `ration_card_type`, `dwelling_type`, `members_count`, `siblings_below_5`, `house_latitude`, `is_deceased` | 0 | Snapshot values. |
| `child_status`, `bmi`, `is_anemic` | 4,311 to 5,651 | One record copied to the other years. |

#### R19. Remove `highest_qualification`
**Priority:** 1

**Finding (Q8, Q9):**
- `highest_qualification` is the latest class of the student. The source copies it into all rows of the student.
- Example: student `096904697373` has the value 5 in all rows, from 2022-23 (class 1) to 2026-27 (class 5).
- In 2024-25, when `highest_qualification` is equal to the class of the row, 30.43% of students are not found in the next year.
- When `highest_qualification` is the class plus 2, only 0.13% are not found.

**Instructions:**
1. Remove `highest_qualification` from the table.
2. If other users need it, rename it to `current_highest_class`.
3. In the data dictionary, mark `current_highest_class` as "snapshot value, not for the model".

**Reason:** This column shows the future class of the student. So it shows directly if the student continued. A model with this column gives false high accuracy.

**Acceptance check:** `highest_qualification` is not in the table.

---

#### R20. Remove `student_age` and keep the date of birth
**Priority:** 1

**Finding (Q8, Q13):**
- `student_age` does not change between years for any student. It is the age on the load date.
- Example: a 2022-23 row shows age 12 for a student born on 2014-12-28.
- In 2025-26 (classes 1 to 10), the date of birth is missing for 169,110 students (2.4%).
- For 45,878 students, the age minus the class is 11 or more. For 21,045 students, it is 2 or less.
- Sample rows show dates on 1 January (for example, 2013-01-01 and 1963-01-01). These are often default values.

**Instructions:**
1. Remove `student_age`, or rename it to `age_on_load_date`.
2. Keep `student_date_of_birth` as a DATE.
3. Add the column `dob_is_estimated`. Set it to TRUE when the source uses a default date.
4. Find the dates of birth that are not possible for a school student. Set them to NULL.
5. Fill the missing dates of birth from the source records, if the rules allow it.

**Reason:** The age in a 2022-23 row must be the age in 2022-23. The ML team calculates this age from the date of birth. Over-age for the class is a strong dropout signal. A wrong date of birth gives a wrong signal.

**Acceptance check:** `student_age` is not in the table, or its name is `age_on_load_date`. Test Q13 shows fewer values of 11 or more and 2 or less.

---

#### R21. Use the school profile of the same academic year
**Priority:** 1

**Finding (Q8, Q11):**
- `total_students` never changes for students who stay in the same school. The school columns are the current profile, copied into all years.
- `functional_status` is NULL for all 2025-26 rows.

**Instructions:**
1. Join the school columns from the school profile (UDISE+) of the same academic year.
2. Add the column `school_status_in_year`. Use only these values: `FUNCTIONAL`, `MERGED`, `CLOSED`.
3. Add the columns `merged_into_school_id` and `merge_or_close_date`.
4. Fill `functional_status`, or remove it.

**Reason:** School mergers and changes in school structure happened between 2022 and 2026. A merged or closed school can cause dropout. The current profile hides this. It also shows wrong numbers of teachers and students for old years.

**Acceptance check:** Test Q8 shows that `total_students` changes for some students who stay in the same school.

---

#### R22. Give household values for each academic year
**Priority:** 1

**Finding (Q8, Q9, Q11):**
- Household columns do not change between years for any student. They are today's values.
- In 2025-26 (classes 1 to 10), 91,377 students show `marital_status` = "Married".
- In 2024-25, 2.59% of "Married" students are not found in the next year. Only 0.11% of "Unmarried" students are not found.

**Instructions:**
1. If the household database keeps history, give the value at the start of each academic year (1 June).
2. If the household database does not keep history, add the prefix `current_` to the column names.
3. Add the column `household_data_as_of_date`.
4. Do this for: `marital_status`, `income_band`, `ration_card_type`, `dwelling_type`, `members_count`, `siblings_below_5`, `house_latitude`, `house_longitude`, `rural_urban_flag`, and the student address columns.

**Reason:** A student who married after leaving school shows "Married" in old rows. This is information from the future. The ML team must know which values are year values. The ML team will not use snapshot values that change after a dropout.

**Acceptance check:** Each household column is a year value, or its name has the prefix `current_`.

---

#### R23. Give the date of death and the verification status
**Priority:** 1

**Finding (Q9, Q11):**
- In 2024-25 (classes 1 to 9), 92,918 rows show "Deceased". 94.95% of these students are not found in the next year.
- In 2025-26 (classes 1 to 10), only 5,760 rows show "Deceased".
- 3,454 rows show "reported deceased". 96.82% of these students have a row in the next year. So most of them are alive and in school.

**Instructions:**
1. Add the columns `date_of_death` and `death_verified_flag`.
2. Keep verified deaths and reports without verification separate.
3. Examine the 2024-25 "Deceased" rows. Find if they come from the placeholder `person_id` (R03).

**Reason:** The ML team removes deceased children from the dropout target. The date of death tells the ML team which academic year to remove. A wrong flag removes real dropouts from the target.

**Acceptance check:** Each "Deceased" row has a date of death and a verification status.

---

#### R24. Join each health record to the academic year of the screening
**Priority:** 2

**Finding (Q8, Q11):**
- `child_status`, `is_anemic`, `child_condition`, and `hb_status` have the same number of NULL values in 2025-26 (4,278,958). So they come from one source (the health screening).
- These columns change between years for only about 4,000 to 6,000 students.

**Instructions:**
1. Add the column `screening_date` to each health record.
2. Put each health record only in the row of the academic year of its `screening_date`.
3. Do not copy one screening into the rows of other years.
4. If a student has two screenings in one year, use the last screening.
5. Write the source of `child_status` in the data dictionary.

**Reason:** A screening from 2025-26 in a 2022-23 row is information from the future. `child_status` = "Left School" is a very strong signal for the model. It is correct only when the ML team knows its date.

**Acceptance check:** Each health record has a `screening_date` in the academic year of its row.

---

#### R25. Give the welfare scheme values for each year, with dates
**Priority:** 2

**Finding (Q9, Q11):**
- `thalliki_vandanam` is TRUE for 3.8% of 2024-25 rows (classes 1 to 9).
- It is TRUE for 71.4% of 2025-26 rows (classes 1 to 10).
- The scheme changed between these years.

**Instructions:**
1. Add the column `benefit_scheme_name` for each year. Examples: `AMMA_VODI`, `TALLIKI_VANDANAM`.
2. Fill the values for 2022-23 and 2023-24 from the scheme of those years, if the source has them.
3. Add the columns `benefit_payment_date`, `benefit_eligible_flag`, and `benefit_ineligibility_reason`.
4. Write the eligibility rules of each scheme in the data dictionary.
5. Write if the eligibility uses attendance.

**Reason:** The model must see the same meaning in the training years and in the current year. The payment date tells the ML team when the value is known. If the eligibility uses attendance, the column repeats attendance information.

**Acceptance check:** Each year has `benefit_scheme_name` and payment dates for the students with a benefit.

---

#### R26. Give the student kit data for all years, with dates
**Priority:** 2

**Finding (Q9):** The kit columns (`uniform_rec` and the others) are NULL for all 2024-25 rows. The sample shows values only in 2025-26 and 2026-27 rows.

**Instructions:**
1. Fill the kit data for all academic years, if the source has it.
2. Add the column `kit_distribution_date`.
3. Use BOOLEAN values: TRUE, FALSE, or NULL.

**Reason:** The model learns from the training years. A column that has values only in the current year cannot help the model.

**Acceptance check:** Each academic year has kit data, or the data dictionary tells why not.

---

### 6.E Attendance

#### R27. Separate "no data" from "zero days"
**Priority:** 1

**Finding (Q6, sample rows):**
- When a student has no attended days, the row also has no working days. Example: in private unaided schools, 93.4% of rows have working days and 93.4% have attended days.
- So the ML team cannot know if the school did not send data or if the student did not come.
- The future months of 2026-27 show 0 working days and 0 attended days.

**Instructions:**
1. Set the working days and the attended days to NULL when the school did not send attendance for the month.
2. Set them to NULL for months that are not complete or not loaded.
3. Set the working days to 0 only when the school was closed for all of the month.
4. If a student is on the attendance register, record the working days, also when the attended days are 0.
5. Keep the current rule that attended days are not more than working days. The audit found no rows that break this rule.

**Reason:** A student on the register with 0 attended days is a strong dropout signal. A school that did not send data gives no information. The two cases need different values.

**Acceptance check:** Test T6 (Section 9). Months after `attendance_as_of_date` are NULL. Students on the register have working days.

---

#### R28. Tell the ML team what the attendance system does when a student stops coming
**Priority:** 1

**Instructions:**
1. Write the rule of the attendance system for a student who stops coming to school.
2. Tell the ML team if the system keeps the student with 0 attended days, or removes the student from the register.
3. If the system removes the student, add the columns `register_removal_date` and `register_removal_reason`.

**Reason:** The removal date is the nearest date to the real dropout date. It helps the label. It also helps the early warning during the year.

**Acceptance check:** The data dictionary gives the rule. The removal columns have values, if the system removes students.

---

#### R29. Add summaries of daily attendance
**Priority:** 2

**Instructions:** If daily attendance data exists, add these columns for each academic year:
1. `first_attended_date`: the first date with attendance.
2. `last_attended_date`: the last date with attendance.
3. `longest_absence_days`: the longest run of working days without attendance.
4. `absence_runs_10_plus`: the number of absence runs of 10 or more working days.

Calculate these values from the start of the year to `attendance_as_of_date`.

**Reason:** A long continuous absence is the earliest sign of dropout. Monthly totals hide it. Example: a student who comes for 12 days and then stops looks like a student with irregular attendance.

**Acceptance check:** The four columns have values for the students with daily attendance.

---

### 6.F Marks and examinations

#### R30. Give the marks of each assessment with the maximum marks
**Priority:** 1

**Finding (Q6, Q10):**
- `fa_marks` and `sa_marks` are totals of many subjects and assessments. Their scale changes between years.
- Median class 10 SA marks: 537 (2022-23), 595 (2023-24), 658 (2024-25), 1,077 (2025-26).
- Median class 1 SA marks: 214, 276, 262, 420 for the same years.
- Some values are not possible: FA marks of 1,035,200 (class 8, 2025-26) and 765,237 (class 4, 2025-26). SA marks of 7,374 in many classes in 2025-26.
- In CBSE schools, only 29.9% of rows have FA marks and 11.0% have SA marks. In ICSE schools, the values are 18.5% and 8.9%.

**Instructions:**
1. Add one set of columns for each assessment: FA1, FA2, FA3, FA4, SA1, and SA2.
2. For each assessment, add `marks_obtained`, `marks_maximum`, `percentage`, `assessment_date`, and `subjects_count`.
3. For other boards, give the assessments of that board, or set the status to `NOT_APPLICABLE` (R31).
4. Remove `fa_marks` and `sa_marks`, or write their rule in the data dictionary.

**Reason:** The model compares students of different classes and years. For this, it needs percentages, not totals. Marks with dates also give a warning during the year, because FA1 is early in the year.

**Acceptance check:** Test T7 (Section 9). All percentages are between 0 and 100.

---

#### R31. Give the status of each assessment
**Priority:** 1

**Finding (Q10):** In 2022-23, the SA marks are 0 for 4.9% to 7.2% of rows. In later years, they are 0 for 1.3% to 6.1% of rows. A value of 0 can mean "absent", "not uploaded", or a real 0.

**Instructions:**
1. For each assessment, add a status column. Use only these values: `WRITTEN`, `ABSENT`, `NOT_UPLOADED`, `EXEMPTED`, `NOT_APPLICABLE`.
2. Set the marks to NULL when the status is not `WRITTEN`.

**Reason:** Absence from an examination is a strong dropout signal. Marks that are not uploaded are not a signal. A value of 0 mixes the two cases.

**Acceptance check:** Each assessment has a status. Marks are NULL when the status is not `WRITTEN`.

---

#### R32. Correct marks that are not possible
**Priority:** 1

**Instructions:**
1. Make sure that `marks_obtained` is between 0 and `marks_maximum`.
2. Find the values that are not possible. Correct them from the source.
3. If you cannot correct a value, set it to NULL.
4. Report the number of corrected values for each year.

**Reason:** Values such as 1,035,200 change the statistics of a full class. They also change the model.

**Acceptance check:** Test T7 (Section 9) returns 0.

---

### 6.G Data quality and standard values

#### R33. Use one value for "no data"
**Priority:** 1

**Finding (Q11, 2025-26, classes 1 to 10):**
- `dwelling_type`: 344,683 rows with the text `'null'` and 1,417,802 rows with NULL.
- `is_anemic`: 640,418 rows with the empty text `''`.
- `child_condition`: 640,418 rows with `''` and 325,128 rows with `'null'`.
- `hb_status`: 1,568,250 rows with `'null'`.
- `fourd_condition`: the sample shows `'-'`.

**Instructions:**
1. Use only the database NULL for "no data".
2. In all text columns, replace these values with NULL: `'null'`, `'NULL'`, `''`, `'-'`, `'--'`, `'NA'`, `'nan'`, `'None'`.

**Reason:** Different "no data" values look like different categories. The model then learns false categories.

**Acceptance check:** Test T5 (Section 9) returns no rows.

---

#### R34. Use one code for each meaning
**Priority:** 1

**Finding (Q9, Q11):**
- `marital_status`: "Single", "Unmarried", and "Single woman" have the same meaning. "Widowed" and "Widower" also have the same meaning. "Separated " has a space at the end.
- `child_status`: "Registerd but Absent" has a spelling error. "Repetition of Child " has a space at the end.
- `student_gender`: "Others" (1 row) and "Transgender" (241 rows).
- `is_deceased`: "Deceased" and "reported deceased" (see R23).

**Instructions:**
1. Remove the spaces at the start and at the end of all text values.
2. Correct the spelling errors.
3. Use one upper-case code for each meaning. Example for `marital_status`: `UNMARRIED`, `MARRIED`, `WIDOWED`, `DIVORCED`, `SEPARATED`.
4. Write the code list of each column in the data dictionary.

**Reason:** If one meaning has different texts, one group becomes many small groups. Small groups give weak and unstable results.

**Acceptance check:** Each text column has only the values in its code list.

---

#### R35. Correct `income_band`
**Priority:** 1

**Finding (Q11, 2025-26, classes 1 to 10):**
- The values include `'>?5L'` (1,529,692 rows), `'?3L–?5L'`, `'?1L–?3L'`, `'?0–?1L'`, `'0–1L'`, `'1L–3L'`, `'3L–5L'`, and `'>5L'`.
- The `?` is a wrong character for the rupee symbol. The dash also has two different forms.
- 6,048,639 students have a BPL ration card. 1,529,692 students have an income above ₹5 lakh. So at least 481,706 students have both. These two values conflict.
- `household_income_category` is NULL for all rows.

**Instructions:**
1. Use UTF-8 encoding in all steps.
2. Use only these codes: `INC_0_1L`, `INC_1L_3L`, `INC_3L_5L`, `INC_ABOVE_5L`.
3. Examine the conflict between BPL ration cards and income above ₹5 lakh.
4. Correct the source, or write the cause in the data dictionary.
5. Fill `household_income_category`, or remove it.

**Reason:** Wrong income values give a wrong risk for poor families. These families need the most help.

**Acceptance check:** `income_band` has only the four codes and NULL. The data dictionary explains the BPL conflict.

---

#### R36. Use the correct data types
**Priority:** 1

**Finding:**
- `school_latitude` and `school_longitude` are STRING.
- `fa_marks` is DECIMAL(38,2) and `sa_marks` is DOUBLE.
- `bmi`, `weight`, and `hb_value` are STRING. `bp_value` has two numbers in one text (for example `110/70`).
- The kit columns use the text "Yes" and "No".
- Private schools show `total_teachers` = 0 and `sanctioned_teachers` = 0 (sample rows).

**Instructions:**
1. Use DOUBLE for all latitude and longitude columns.
2. Use the same number type for all marks columns.
3. Use DOUBLE for `bmi`, `weight`, and `hb_value`.
4. Split `bp_value` into `bp_systolic` and `bp_diastolic`.
5. Use BOOLEAN for all yes/no columns.
6. Use DATE for all date columns.
7. Use NULL, not 0, when a number is not known. Example: the teachers of private schools.

**Reason:** Correct types prevent conversion errors. A 0 for "not known" looks like a real value. Example: 0 teachers looks like a school with no teachers.

**Acceptance check:** The table schema shows the types above. Private schools without teacher data show NULL.

---

#### R37. Correct the coordinates
**Priority:** 2

**Finding:** The sample has house coordinates missing for 5 of 20 rows.

**Instructions:**
1. Set house and school coordinates to NULL when they are outside Andhra Pradesh.
2. Fill the missing house coordinates from the household geo-tag source, if available.
3. Add the column `house_coordinate_source`.

**Reason:** The model uses the distance from home to school. Wrong coordinates give wrong distances.

**Acceptance check:** All coordinates are inside Andhra Pradesh or NULL.

---

#### R38. Fill the missing values of key columns
**Priority:** 2

**Finding (Q11, 2025-26, classes 1 to 10):**
- `social_category`: NULL for 1,109,238 rows (15.6%).
- `ration_card_type`: NULL for 786,957 rows.
- `rural_urban_flag`: NULL for 39,965 rows.
- `medium_name`: NULL for 27,329 rows.
- `student_gender`: NULL for 13,956 rows.

**Instructions:**
1. Fill these values from the source systems.
2. If you cannot fill a value, keep NULL.
3. Do not put a default value in place of a missing value.

**Reason:** Missing values reduce the information about these students. A default value adds false information.

**Acceptance check:** The NULL counts are lower than the counts above.

---

### 6.H New information

#### R39. Add data about children with special needs
**Priority:** 2

**Instructions:**
1. Add the columns `cwsn_flag` and `disability_type` for each academic year.
2. Use UDISE+ as the source.

**Reason:** Children with special needs have a higher dropout risk. Now the table shows this only through the school type.

**Acceptance check:** The two columns have values from UDISE+ for each year.

---

#### R40. Add parent data
**Priority:** 2

**Finding:** `guardian_person_id` is NULL in all sample rows.

**Instructions:**
1. Add the columns `mother_education`, `father_education`, `mother_occupation`, and `father_occupation`.
2. Add the columns `mother_alive_flag` and `father_alive_flag`, with the date of death if applicable.
3. Add the column `guardian_relation` when the child lives with a guardian.
4. Fill `guardian_person_id`, or remove it.

**Reason:** The education of the parents and the death of a parent are known dropout risk factors.

**Acceptance check:** The new columns have values for most students.

---

#### R41. Add seasonal migration of the family
**Priority:** 3

**Instructions:** If a source records the seasonal migration of the family for work, add `family_migration_flag` and the migration period.

**Reason:** Seasonal migration for work takes children out of school.

**Acceptance check:** The column has values, or the data team tells the ML team that no source exists.

---

### 6.I Delivery and operations

#### R42. Write a data dictionary
**Priority:** 1

**Instructions:** For each column, write:
1. The meaning.
2. The source system.
3. The data type.
4. The allowed values.
5. The meaning of NULL.
6. The type of value: year value or snapshot value.
7. The as-of date.
8. The refresh frequency.

**Reason:** The ML team must know which columns are correct for each year. Without this, the ML team can use a snapshot value by mistake.

**Acceptance check:** The data dictionary has all 8 items for each column.

---

#### R43. Give an as-of date for each source
**Priority:** 1

**Finding:** `load_date` has one value (2026-10-08) for all rows.

**Instructions:** Add these columns:
1. `enrolment_as_of_date`
2. `attendance_as_of_date`
3. `marks_as_of_date`
4. `household_as_of_date`
5. `health_as_of_date`
6. `school_profile_as_of_date`

**Reason:** The sources update at different times. The model must know the age of each value.

**Acceptance check:** Each row has the six dates.

---

#### R44. Make frozen copies for training
**Priority:** 1

**Instructions:**
1. When the ML team asks, make a frozen copy of the table.
2. Put the date in the name of the copy. Example: `final_student_dropout_info_v2_snap_20261115`.
3. Do not change a frozen copy.
4. Write all corrections to past years in the change log (R45).

**Reason:** A government audit can ask the ML team to repeat a model result. This is possible only with the same data.

**Acceptance check:** The frozen copy has the same row count and the same values as on its date.

---

#### R45. Keep a change log and tell the ML team before changes
**Priority:** 1

**Instructions:**
1. Tell the ML team before you change a rule, a source, or the meaning of a column.
2. Write each change in a change log, with the date and the reason.

**Reason:** A change without notice can break the model without an error message.

**Acceptance check:** The change log exists and has an entry for each change in this request.

---

#### R46. Make a quality report after each load
**Priority:** 1

**Instructions:** After each load, make a report with:
1. The number of rows and students for each year, class, and school management.
2. The number of duplicate rows (Test T1).
3. The percentage of rows with attendance and with marks, for each month.
4. The number of values outside the code lists.
5. A comparison of the enrolment numbers with the UDISE+ numbers.

**Reason:** The report finds problems before they get to the model.

**Acceptance check:** The data team sends the report to the ML team after each load.

---

## 7. Current columns and the action for each column

| Current column(s) | Action | Requirement |
|---|---|---|
| `person_id` | Change: remove the placeholder ID, give one permanent key, keep as STRING | R03, R04, R05 |
| `student_id` | Keep as STRING. Keep the ID of each year | R04, R05 |
| `academic_year` | Keep. Use as the partition key | R01 |
| `education_level` | Keep | — |
| `highest_qualification` | **Remove** (data leakage) | R19 |
| `is_dropout`, `dropout_reason` | Change: write the rule, or rename with `source_` | R15 |
| `class_currently_enrolled` | Change: one row for each year, code list, correct class 0 and class 13 | R02, R07 |
| `medium_name`, `school_medium` | Keep. Write the difference in the data dictionary | R42 |
| `thalliki_vandanam` | Change: add scheme name, dates, and history | R25 |
| `student_age` | **Remove**, or rename to `age_on_load_date` | R20 |
| `student_date_of_birth` | Change: add `dob_is_estimated`, correct the values that are not possible | R20 |
| `student_gender`, `social_category` | Change: code list, fill missing values | R34, R38 |
| `dwelling_type`, `ration_card_type`, `members_count`, `siblings_below_5`, `rural_urban_flag` | Change: year value or prefix `current_`, one NULL value | R22, R33 |
| `marital_status` | Change: code list, year value or prefix `current_` | R22, R34 |
| `income_band` | Change: encoding, code list, examine the BPL conflict | R35 |
| `household_income_category` | Fill or remove (NULL for all rows) | R35 |
| `is_deceased` | Change: add date of death and verification status | R23 |
| `student_district_id`, `student_district_name`, `student_mandal_id`, `student_mandal_name`, `student_village_name`, `secretariat_code` | Change: year value or prefix `current_` | R22 |
| `household_id`, `father_person_id`, `mother_person_id` | Keep as STRING | R05 |
| `guardian_person_id` | Fill or remove (NULL in all sample rows) | R40 |
| `religion`, `caste_name` | Keep. These are not model inputs. The ML team uses them only for fairness checks | — |
| `domicile_status`, `govt_employment_type` | Examine. Fill or remove (NULL in all sample rows) | R42 |
| `house_latitude`, `house_longitude` | Change: range check, source, year value or prefix `current_` | R22, R37 |
| `school_id`, `school_name`, `canonical_facility_id` | Keep | — |
| `school_category`, `school_management`, `board_affiliation`, `school_type_coed_boys_girls`, `school_in_rural_urban`, `lowest_class`, `highest_class` | Change: use the profile of the same year | R21 |
| `school_district_code` to `school_village_name` | Change: use the profile of the same year | R21 |
| `school_latitude`, `school_longitude` | Change: DOUBLE, range check | R36, R37 |
| `functional_status` | Fill or remove (NULL for all rows) | R21 |
| `total_students`, `total_boys`, `total_girls`, `total_teachers`, `sanctioned_teachers`, `student_teacher_ratio`, `total_classrooms`, `classrooms_good_condition`, `building_status` | Change: profile of the same year, NULL in place of 0 for "not known" | R21, R36 |
| `has_electricity` to `internet_available` (13 facility columns) | Change: profile of the same year | R21 |
| `child_status` | Change: screening date, code list, spelling | R24, R34 |
| `bmi`, `hb_value`, `bp_value`, `weight`, `vision`, `hearing`, `is_anemic`, `child_condition`, `fourd_condition`, `hb_status` | Change: screening date, numeric types, one NULL value | R24, R33, R36 |
| `bags_rec` to `practical_records_rec` (8 kit columns) | Change: all years, distribution date, BOOLEAN | R26, R36 |
| `jan_attended` to `dec_working_days` (24 columns) | Change: NULL for "no data", load on time | R09, R27 |
| `fa_marks`, `sa_marks` | Change: replace with marks for each assessment | R30, R31, R32 |
| `load_date` | Change: add an as-of date for each source | R43 |

---

## 8. New columns

| New column(s) | Requirement |
|---|---|
| `person_id_status` | R03 |
| `enrolments_in_year`, `first_school_id_in_year`, `school_change_date` | R02 |
| `student_status_at_year_end`, `student_status_date`, `leaving_reason` | R12 |
| `record_source`, `record_created_date`, `record_last_updated_date`, `school_confirmed_flag` | R13 |
| `tc_issued_flag`, `tc_issue_date`, `tc_to_school_id`, `admission_date` | R14 |
| `ssc_result`, `ssc_result_date`, `supplementary_result`, `supplementary_result_date`, `ssc_attempt_number` | R16 |
| `institution_type`, `institution_id` | R17 |
| `admission_confirmed_flag` | R18 |
| `dob_is_estimated` | R20 |
| `school_status_in_year`, `merged_into_school_id`, `merge_or_close_date` | R21 |
| `household_data_as_of_date` | R22 |
| `date_of_death`, `death_verified_flag` | R23 |
| `screening_date` | R24 |
| `benefit_scheme_name`, `benefit_payment_date`, `benefit_eligible_flag`, `benefit_ineligibility_reason` | R25 |
| `kit_distribution_date` | R26 |
| `register_removal_date`, `register_removal_reason` | R28 |
| `first_attended_date`, `last_attended_date`, `longest_absence_days`, `absence_runs_10_plus` | R29 |
| For each of FA1, FA2, FA3, FA4, SA1, SA2: `marks_obtained`, `marks_maximum`, `percentage`, `assessment_date`, `subjects_count`, `status` | R30, R31 |
| `bp_systolic`, `bp_diastolic` | R36 |
| `house_coordinate_source` | R37 |
| `cwsn_flag`, `disability_type` | R39 |
| `mother_education`, `father_education`, `mother_occupation`, `father_occupation`, `mother_alive_flag`, `father_alive_flag`, `guardian_relation` | R40 |
| `family_migration_flag` | R41 |
| `enrolment_as_of_date`, `attendance_as_of_date`, `marks_as_of_date`, `household_as_of_date`, `health_as_of_date`, `school_profile_as_of_date` | R09, R43 |

---

## 9. Acceptance tests

Run these tests on the new table before you give it to the ML team. Also run the audit queries Q2, Q3, Q8, Q12, and Q13 again, and send the results.

In the tests, `<NEW_TABLE>` is the name of the new table.

**T1. One row for each student for each year (R02).** Expected result: no rows.
```sql
SELECT academic_year, person_id, COUNT(*) AS n_rows
FROM <NEW_TABLE>
WHERE person_id IS NOT NULL
GROUP BY academic_year, person_id
HAVING COUNT(*) > 1
LIMIT 100;
```

**T2. No placeholder IDs (R03).** Expected result: `n_rows` is not more than the number of academic years.
```sql
SELECT person_id, COUNT(*) AS n_rows, COUNT(DISTINCT student_id) AS n_student_ids
FROM <NEW_TABLE>
WHERE person_id IS NOT NULL
GROUP BY person_id
ORDER BY n_rows DESC
LIMIT 20;
```

**T3. Identity conflicts (R04).** Expected result: lower counts than in the audit (68,088 and 201,034). Each remaining type of conflict has a reason.
```sql
SELECT 'student_id_with_multiple_person_ids' AS check_name, COUNT(*) AS n
FROM (SELECT student_id FROM <NEW_TABLE>
      WHERE student_id IS NOT NULL AND person_id IS NOT NULL
      GROUP BY student_id HAVING MIN(person_id) <> MAX(person_id)) x
UNION ALL
SELECT 'person_id_with_multiple_student_ids', COUNT(*)
FROM (SELECT person_id FROM <NEW_TABLE>
      WHERE student_id IS NOT NULL AND person_id IS NOT NULL
      GROUP BY person_id HAVING MIN(student_id) <> MAX(student_id)) y;
```

**T4. Completeness of 2023-24 (R08).** Expected result: the 2023-24 counts are similar to the other years.
```sql
SELECT academic_year, class_currently_enrolled AS cls, COUNT(*) AS n_rows
FROM <NEW_TABLE>
WHERE class_currently_enrolled BETWEEN -3 AND 2
GROUP BY academic_year, class_currently_enrolled
ORDER BY cls, academic_year;
```

**T5. One value for "no data" (R33).** Expected result: no rows. Add the other text columns in the same way.
```sql
SELECT 'dwelling_type' AS col, dwelling_type AS value, COUNT(*) AS n FROM <NEW_TABLE>
WHERE TRIM(LOWER(dwelling_type)) IN ('null','','-','--','na','nan','none') GROUP BY 2
UNION ALL
SELECT 'is_anemic', is_anemic, COUNT(*) FROM <NEW_TABLE>
WHERE TRIM(LOWER(is_anemic)) IN ('null','','-','--','na','nan','none') GROUP BY 2
UNION ALL
SELECT 'child_condition', child_condition, COUNT(*) FROM <NEW_TABLE>
WHERE TRIM(LOWER(child_condition)) IN ('null','','-','--','na','nan','none') GROUP BY 2
UNION ALL
SELECT 'hb_status', hb_status, COUNT(*) FROM <NEW_TABLE>
WHERE TRIM(LOWER(hb_status)) IN ('null','','-','--','na','nan','none') GROUP BY 2;
```

**T6. Attendance rules (R27).** Expected result: 0 in both columns. The example uses August. Do the same test for each month.
```sql
SELECT academic_year,
       SUM(CASE WHEN aug_attended > aug_working_days THEN 1 ELSE 0 END) AS attended_gt_working,
       SUM(CASE WHEN CAST(CONCAT(SUBSTR(academic_year, 1, 4), '-08-31') AS TIMESTAMP)
                     > CAST(attendance_as_of_date AS TIMESTAMP)
                 AND aug_working_days IS NOT NULL THEN 1 ELSE 0 END) AS future_month_not_null
FROM <NEW_TABLE>
GROUP BY academic_year
ORDER BY academic_year;
```

**T7. Marks range (R30, R32).** Expected result: 0. Do the same test for each assessment.
```sql
SELECT COUNT(*) AS n_out_of_range
FROM <NEW_TABLE>
WHERE fa1_percentage < 0 OR fa1_percentage > 100
   OR fa1_marks_obtained > fa1_marks_maximum;
```

**T8. Status coverage (R12).** Expected result: `NOT_KNOWN` and NULL are less than 5% of each completed year.
```sql
SELECT academic_year, student_status_at_year_end, COUNT(*) AS n_rows
FROM <NEW_TABLE>
WHERE class_currently_enrolled BETWEEN 1 AND 10
GROUP BY academic_year, student_status_at_year_end
ORDER BY academic_year, n_rows DESC;
```

---

## 10. Questions for the data team

Please answer these questions with the new table.

1. What is the `person_id` value with 127,571 rows in 2024-25? Why does it exist? (R03)
2. Why does 2023-24 have only 38,129 class 1 rows? (R08)
3. What do the class codes -3, -2, -1, 0, and 13 mean? (R07)
4. Does the source system copy students into the new academic year automatically? On which date? (R13)
5. What does the attendance system do when a student stops coming to school? (R28)
6. What is the source of `child_status` and of the health columns? On which date was each record made? (R24)
7. Which rule sets `is_dropout` and `dropout_reason`? (R15)
8. Does the household database keep the history of its values? (R22)
9. Does the source have the marks of each assessment (FA1 to FA4, SA1, SA2) with the maximum marks? (R30)
10. Which of these sources are available: BIE, ITI admissions, SBTET, APOSS, NIOS, SSDC? (R17)
11. Does the source have daily attendance? (R29)
12. Does the source have 2021-22 data? (R11)
13. Is the class 10 examination result (SSC) available for each year? (R16)

---

## 11. Order of work

Do the work in this order. Send the table to the ML team after each step. Then the ML team can start the work in parallel.

| Step | Requirements | Result |
|---|---|---|
| 1 | R01, R02, R03, R04, R05, R06, R07, R08 | Correct identity and complete rows. |
| 2 | R12, R13, R15, R16, R17, R27, R28 | Correct facts for the label. |
| 3 | R19, R20, R21, R22, R23, R30, R31, R32, R33, R34, R35, R36 | No data leakage and clean values. |
| 4 | R09, R42, R43, R44, R45, R46 | Correct operations. |
| 5 | All Priority 2 and Priority 3 requirements | More information for the model. |
