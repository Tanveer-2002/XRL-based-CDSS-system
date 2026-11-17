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

## 2. General Classification Performance (Standard Metrics)

| Model | Accuracy (%) | Balanced Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Historical Clinician | 53.26 | 60.84 | 70.69 | 60.84 | 0.6084 | 0.5197 |
| XGBoost | 17.06 | 30.53 | 10.52 | 30.53 | 0.1228 | 0.107 |
| Behavioral Cloning (BC) | 41.8 | 46.25 | 50.03 | 46.25 | 0.4055 | 0.3175 |
| Discrete BCQ | 38.15 | 46.45 | 40.01 | 46.45 | 0.3328 | 0.2715 |
| Discrete CQL (Ours) | 42.45 | 50.65 | 37.89 | 50.65 | 0.4057 | 0.2934 |

---

## 3. Comprehensive Clinical Decision & Offline RL Evaluation

| Model | Category | Guideline Acc (%) | Balanced Acc (%) | Clinician Match (%) | Macro F1 | Sensitivity (%) | False Alarm (%) | False / Pt-Yr | Median Lead Time (d) | Adjacent Acc (%) | Mean Return (V) | FQE Value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Historical Clinician | Empirical Baseline | 53.26 | 60.84 | 100.0 | 0.6084 | 100.0 | 0.0 | 0.0 | 0.0 | 96.35 | 13.25 | 3.08 |
| XGBoost | Supervised ML | 17.06 | 30.53 | 23.05 | 0.1228 | 99.35 | 69.51 | 1.85 | 230.0 | 47.4 | -6.11 | 1.76 |
| Behavioral Cloning (BC) | Imitation Learning | 41.8 | 46.25 | 60.42 | 0.4055 | 66.23 | 5.79 | 0.15 | 0.0 | 89.84 | 12.69 | 3.71 |
| Discrete BCQ | Offline RL (Constraint) | 38.15 | 46.45 | 48.57 | 0.3328 | 87.01 | 35.95 | 0.96 | 0.0 | 91.28 | 7.92 | 3.45 |
| Discrete CQL (Ours) | Offline RL (Conservative Q) | 42.45 | 50.65 | 59.64 | 0.4057 | 73.38 | 10.9 | 0.29 | 0.0 | 92.19 | 13.15 | 3.48 |

---

## 4. Metric Descriptions & Interpretation
1. **Clinical Guideline Accuracy (%)**: Percentage of encounters where the model's recommendation perfectly matched the Gold-Standard American Diabetes Association (ADA) clinical rules.
2. **Balanced Accuracy (%)**: Macro-averaged recall across all 4 monitoring classes ($\frac{1}{\text{C}}\sum \text{Recall}_c$), preventing majority class bias.
3. **Macro Precision (%)**: Unweighted average precision across all 4 classes, measuring positive predictive fidelity without majority skew.
4. **Macro Recall (%)**: Unweighted average recall across all 4 classes, equivalent to Balanced Accuracy.
5. **Macro F1-Score**: Unweighted harmonic mean of precision and recall across all 4 action classes, penalizing models that fail on rare critical alert actions.
6. **Weighted F1-Score**: F1-score weighted by class prevalence in the clinical test cohort.
7. **Early Deterioration Sensitivity (%)**: Percentage of deteriorating or rapidly deteriorating encounters where an active monitoring or clinical alert was raised.
8. **Alarm Fatigue False Alarm Rate (%)**: Percentage of stable encounters with unwarranted high-urgency alerts.
9. **False Alerts per Patient-Year**: Clinically normalized frequency of false alarms per 365.25 days of patient observation (lower is better to prevent alarm fatigue).
10. **Median Early Warning Lead Time (Days)**: Median days in advance of acute deterioration onset that the model first escalated monitoring.
11. **Action Adjacency Concordance (%)**: Percentage of predictions within $\pm 1$ tier of the clinical guideline ($|a_{\text{pred}} - a_{\text{true}}| \le 1$), evaluating near-miss safety.
12. **Mean Trajectory Return ($V^\pi$)**: Cumulative discounted reward ($\sum_{t=0}^T \gamma^t r_t$) accumulated across patient episodes under the learned policy.
13. **Fitted Q-Evaluation (FQE Value)**: Off-Policy Evaluation (OPE) estimate of expected return from initial visit states ($s_0$), evaluating value without importance sampling variance.

---

## 5. Visual Artifacts
- General Classification Metrics: `general_metrics_comparison.png`
- 8-Panel Comparison Bar Charts: `model_benchmark_comparison.png`
- Confusion Matrix Grid: `confusion_matrices_all.png`
