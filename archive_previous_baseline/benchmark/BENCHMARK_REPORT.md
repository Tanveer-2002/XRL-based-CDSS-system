# Multi-Model Benchmark Report: Type 2 Diabetes CDSS

This report documents the empirical evaluation and benchmarking of the primary **Discrete CQL (Conservative Q-Learning)** Clinical Decision Support System against established machine learning and offline reinforcement learning baselines on the **MIMIC-IV** cohort.

---

## 1. Experimental Methodology
- **Cohort**: MIMIC-IV Type 2 Diabetes Trajectory Dataset (500 patients, 3,888 total encounters).
- **Split Strategy**: Strict patient-level partition (`split_by_patient`) with 80% Train (400 patients, 3,099 transitions) and 20% Test (100 patients, 789 transitions). Fixed random seed `42` ensures zero cross-patient data leakage.
- **State Representation**: 56-dimensional clinical biomarker vector constructed by `StateConstructor` (scaled biomarkers, 30-day change rates, rolling moving averages, trend direction codes, demographics).
- **Action Space**: 4 discrete monitoring frequencies defined by ADA guidelines:
  - `0: ROUTINE_MONITORING` (Standard 6-month checkup)
  - `1: INCREASED_MONITORING` (1-3 month close monitoring)
  - `2: EARLY_RISK_ALERT` (2-4 week urgent flag for deteriorating metabolic status)
  - `3: CLINICAL_EVALUATION` (Immediate comprehensive specialist review)

---

## 2. Side-by-Side Model Comparison Table

| Model | Category | Guideline Acc (%) | Balanced Acc (%) | Clinician Match (%) | Macro F1 | Sensitivity (%) | False Alarm (%) | False / Pt-Yr | Median Lead Time (d) | Adjacent Acc (%) | Mean Return (V) | FQE Value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Historical Clinician | Empirical Baseline | 53.75 | 64.06 | 100.0 | 0.6391 | 100.0 | 0.0 | 0.0 | 0.0 | 94.47 | 13.76 | 3.24 |
| XGBoost | Supervised ML | 21.34 | 29.06 | 22.92 | 0.1378 | 100.0 | 76.19 | 1.98 | 178.0 | 43.48 | -7.68 | 2.31 |
| Behavioral Cloning (BC) | Imitation Learning | 44.27 | 44.59 | 63.24 | 0.4046 | 71.19 | 3.7 | 0.1 | 0.0 | 88.54 | 13.45 | 3.24 |
| Discrete BCQ | Offline RL (Constraint) | 36.36 | 43.31 | 43.87 | 0.2997 | 88.14 | 37.04 | 0.96 | 0.0 | 89.33 | 7.35 | 3.01 |
| Discrete CQL (Ours) | Offline RL (Conservative Q) | 36.76 | 30.57 | 56.92 | 0.2222 | 18.64 | 1.06 | 0.03 | 0.0 | 79.05 | 8.07 | 3.03 |

---

## 3. Metric Descriptions & Interpretation
1. **Clinical Guideline Accuracy (%)**: Percentage of encounters where the model's recommendation perfectly matched the Gold-Standard American Diabetes Association (ADA) clinical rules.
2. **Balanced Accuracy (%)**: Macro-averaged recall across all 4 monitoring classes ($\frac1{C}\sum \text{Recall}_c$), preventing majority class bias.
3. **Macro F1-Score**: Unweighted harmonic mean of precision and recall across all 4 action classes, penalizing models that fail on rare critical alert actions.
4. **Early Deterioration Sensitivity (%)**: Percentage of deteriorating or rapidly deteriorating encounters where an active monitoring or clinical alert was raised.
5. **Alarm Fatigue False Alarm Rate (%)**: Percentage of stable encounters with unwarranted high-urgency alerts.
6. **False Alerts per Patient-Year**: Clinically normalized frequency of false alarms per 365.25 days of patient observation (lower is better to prevent alarm fatigue).
7. **Median Early Warning Lead Time (Days)**: Median days in advance of acute deterioration onset that the model first escalated monitoring.
8. **Action Adjacency Concordance (%)**: Percentage of predictions within $\pm 1$ tier of the clinical guideline ($|a_{pred} - a_{true}| \le 1$), evaluating near-miss safety.
9. **Mean Trajectory Return ($V^\pi$)**: Cumulative discounted reward ($\sum_{t=0}^T \gamma^t r_t$) accumulated across patient episodes under the learned policy.
10. **Fitted Q-Evaluation (FQE Value)**: Off-Policy Evaluation (OPE) estimate of expected return from initial visit states ($s_0$), evaluating value without importance sampling variance.

---

## 4. Visual Artifacts
- 8-Panel Comparison Bar Charts: `model_benchmark_comparison.png`
- Confusion Matrix Grid: `confusion_matrices_all.png`
