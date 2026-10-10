# Effects of Model Training Without the Staging Table Changes

| Item | Value |
|---|---|
| Table | `aiml_usecase_curated.final_student_dropout_info_new` |
| From | AI/ML team, Student Dropout System |
| Date | 2026-10-09 |
| Related document | `staging_table_change_request.md` (requirements R01 to R46) |
| Writing standard | ASD-STE100 (Simplified Technical English) |

---

## 1. Summary

- We can train a model without the changes. But the model will be less accurate, and **we cannot measure its real accuracy.**
- Our code can work around some problems. Each workaround has a cost.
- Our code cannot correct the other problems, because only the source data has the correct information.
- **The biggest problem is the label.** Without the changes, we cannot know which students really dropped out.

---

## 2. Problems that our code can work around

| No. | Problem | Workaround in our code | Remaining cost |
|---|---|---|---|
| 1 | `highest_qualification` shows the future class. | We remove the column. | None for the model. Other users of the table can still use it by mistake. |
| 2 | `student_age` is the age today. | We calculate the age from the date of birth. | 169,110 students (2.4%) have no date of birth. Some dates are default values. Their over-age signal is wrong or missing. |
| 3 | The placeholder `person_id` has 127,571 rows in 2024-25. | We remove all rows of this ID. | We lose 127,571 students. If they come from specific schools or districts, the model is weak for those groups. |
| 4 | Some students have two or three rows in one year. | We keep one row with a fixed rule. | Our rule can select the wrong class or school. |
| 5 | "No data" has many forms (`'null'`, `''`, `'-'`). Some values have spelling errors. | We clean the values in our code. | None for the model. The dashboards and other users still see dirty values. |
| 6 | FA and SA totals have a different scale each year. Some values are not possible. | We use the rank of the student inside the class and year. We remove values that are not possible. | We cannot tell "absent from the examination" from "marks not uploaded". The rank hides real changes between years. |
| 7 | 2022-23 has no attendance for June and July. | We calculate the first quarter of 2022-23 from August only. | The 2022-23 first quarter is different from the other years. |

---

## 3. Problems that our code cannot correct

### 3.1 We cannot know who really dropped out (the label)

**Problem:**
- The source system keeps students in the table after they leave school.
- So "no row in the next year" is not a dropout signal. For 2024-25 it finds only 0.09% to 0.34% of students.
- The only other signal is "no attendance in the next year". But each year this signal marks 8.9% to 12.5% of students who attended this year. That is about 600,000 students each year.

**What our code can do:** We can use "no attendance in the next year" as the label. We can also check if the school sent attendance for its other students.

**What stays wrong:**
1. We cannot separate a dropout from a transfer to a school outside the table.
2. We cannot separate a dropout from a migration out of the state.
3. We cannot separate a dropout from a death without a correct date.
4. We cannot separate "the student left" from "the school did not send the attendance of this student". Both cases show no working days.
5. We do not know how many of the 600,000 students each year are real dropouts. Probably a large part are not.

**Effect:**
- The model learns a mix of dropouts, transfers, migrations, and data gaps.
- The risk list will contain many students who did not drop out.

**Requirements that correct this problem:** R12, R13, R14, R23, R27, R28.

### 3.2 We cannot measure the real accuracy of the model

**Problem:** The test year has the same wrong label as the training years.

**Effect:**
1. The test result shows how well the model predicts the wrong label. It does not show how well the model finds real dropouts.
2. A high test score can be false. A low test score can also be false.
3. We cannot give the Government a reliable accuracy number.
4. We cannot compare two models correctly, because both are measured against the wrong label.

**Requirements that correct this problem:** R12, R13, R27, R28.

### 3.3 One year of training data is mostly wrong

**Problem:** 2023-24 has rows missing. It has only 38,129 class 1 rows; other years have about 600,000. Of the 122,488 students of 2022-23 "not found" in 2023-24, 91,688 (74.9%) come back in 2024-25.

**Effect:**
1. About 3 of 4 "dropouts" from 2022-23 are false. We must remove 2022-23 from training, which leaves fewer training years.
2. The history of many students is wrong. Example: about 600,000 class 1 students of 2023-24 have no 2023-24 row. In 2024-25 they look like "new" students or "rejoiners".
3. "Rejoiner" and "new student" are strong risk signals, so the model learns these signals from wrong data.

**Requirement that corrects this problem:** R08.

### 3.4 Class 10 is weak or not possible

**Problem:**
- 176,473 (21.5%) of the 2025-26 class 10 students show class 10 again in 2026-27.
- We cannot know if these students failed the SSC examination or if the system copied their rows.
- The table does not have ITI, polytechnic, open school, or skill training data. Students who join these institutions look like dropouts.

**Effect:**
1. We must remove class 10 from the model, or make a separate model with a weak label.
2. Class 10 is a high-risk class. The Government will get no list, or a weak list, for class 10.
3. The model can mark ITI and polytechnic students as dropouts. Field staff will then visit students who are in training.

**Requirements that correct this problem:** R16, R17, R18.

### 3.5 We must remove strong information

To prevent data leakage, we must remove the columns that show today's values in old rows.

| Information that we lose | Why it is important |
|---|---|
| `child_status` (for example "Left School", "Registerd but Absent") | Very strong signal: 6.54% of "Left School" students are not found next year, against 0.27% of "Present" students. Without its date, it gives data leakage. |
| `marital_status` | Child marriage is a known cause of dropout for girls. The current value is from today, not from the year of the row. |
| Health data (anaemia, condition) | One screening is copied into all years, so the data is not correct for old years. |
| Welfare benefit | The scheme changed. 3.8% of students had the benefit in 2024-25 and 71.4% in 2025-26. The training years and the current year are different. |
| Student kits | NULL for all 2024-25 rows. The model cannot learn from a column that is empty in the training years. |
| Income band | At least 481,706 students have a BPL ration card and income above ₹5 lakh. We cannot trust the values. |
| SSC result, daily attendance, assessment marks with dates | These data are not in the table. |

**Effect:** The model has less information, so it finds fewer real dropouts.

**Requirements that correct this problem:** R16, R22, R24, R25, R26, R29, R30, R35.

### 3.6 School and household values are from today

**Problem:** The school profile and the household data are the current values, copied into all years.

**Effect:**
1. If a school merged or closed between 2022 and 2026, the model does not see it. A school closure can cause dropout.
2. Teacher and student numbers for old years are wrong.
3. If a family moved house, the distance from home to school is wrong for the old years.
4. We can keep the values that change slowly (ration card, dwelling). But they contain small errors and a small amount of data leakage.

**Requirements that correct this problem:** R21, R22.

### 3.7 The early-warning list is late

**Problem:** On 2026-10-08, the table had 2026-27 attendance only for June and July.

**Effect:**
1. A student who stopped coming in August or September is not on the list.
2. Field staff act 2 to 3 months late.
3. Our code cannot correct this, because the data is not in the table.

**Requirement that corrects this problem:** R09.

### 3.8 Some groups of students get wrong scores

**Problem:** Some school types have many rows with no attendance and no marks (2025-26):

| School type | Rows with no attendance and no marks |
|---|---|
| NCLP | 66.1% |
| CENTRAL GOVT. | 38.5% |
| MADARSA UNRECOGNIZED | 25.8% |
| PVT.UNAIDED (CBSE SYLLABUS) | 15.0% |
| PVT.UNAIDED (ICSE SYLLABUS) | 13.4% |

**Effect:**
1. The model can give a high risk to all students of these schools, because their data is missing.
2. This gives many false alarms for these schools.
3. CBSE and ICSE students have marks for only 9% to 30% of rows, so the model is weaker for them.
4. NCLP children (former child labourers) are a high-risk group, so wrong scores for them are a serious problem.

**Requirements that correct this problem:** R27, R28, R30, R31.

### 3.9 We cannot repeat or defend a result

**Problem:** The table has no frozen copies, no as-of dates for each source, and no change log.

**Effect:**
1. If the table changes, we cannot repeat a model result.
2. In a Government audit, we cannot show which data the model used.
3. We cannot prove that each value was known at the time of the prediction.
4. A silent change in the table can break the model without an error message.

**Requirements that correct this problem:** R42, R43, R44, R45.

### 3.10 We do the work two times

**Problem:** The data team will make these changes later.

**Effect:**
1. We must train, test, and calibrate the model again.
2. The risk scores of many students will change.
3. Officials will see a different list for the same students, so they can stop trusting the system.

---

## 4. Total effect on the model

1. **Lower accuracy.** The label has much noise, so even the best algorithm cannot learn the real pattern well.
2. **Unknown accuracy.** We cannot measure the real accuracy (see 3.2).
3. **Many false alarms.** The list contains transfers, migrations, and students with missing data.
4. **Missed students.** Class 10 is weak or missing. Students who stopped in August or September are missing.
5. **Unfair scores.** Groups with more missing data get wrong scores (see 3.8).

---

## 5. Effect on the Government use

1. **Wasted field visits.** Field staff visit many students who did not drop out.
2. **Lost trust.** After many wrong visits, field staff stop using the list.
3. **Children not helped.** Real dropouts that the model misses get no help.
4. **Wrong official numbers.** The table gives a "not found" rate of 0.09% to 0.64% for recent years. A dashboard with these numbers shows a dropout problem that is too small.
5. **Audit risk.** The ML team cannot defend a prediction or repeat a result.

---

## 6. Recommendation

1. **We can start now with a pilot model (version 1).** It uses all the workarounds in Section 2 and the attendance label from Section 3.1.
2. **Do not use the pilot model for official dropout numbers or for a statewide rollout.** Use it only to test the pipeline and to get feedback from a small number of districts.
3. **Before production, get the minimum set of changes:**
   - R03: remove the placeholder `person_id`.
   - R08: load the missing 2023-24 rows.
   - R12 and R13: add the official status, and mark carried-forward rows. If this is not possible, R27 and R28 (attendance rules).
   - R16 and R17: add the class 10 result and the ITI and polytechnic rows.
   - R44: make frozen copies of the table.
4. **When these changes arrive, train the production model (version 2).** Compare it with version 1 and report the differences.
