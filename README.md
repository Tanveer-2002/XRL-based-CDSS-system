# Type 2 Diabetes Clinical Decision Support System (CDSS)

An advanced clinical decision support system designed to continuously analyze longitudinal health trajectories, detect early metabolic and cardiovascular risk, select safe monitoring actions using Discrete Conservative Q-Learning (CQL Offline RL), provide Explainable AI (SHAP) explanations for clinical transparency, and generate cumulative health reports with an interactive dashboard and REST API.

---

## 1. System Architecture & Workflow

Unlike conventional healthcare systems that evaluate a patient from a single report or isolated visit, this CDSS focuses on **longitudinal health trajectory analysis**, combining multiple observations over time to determine whether important clinical indicators are improving, remaining stable, or deteriorating.

```mermaid
flowchart LR
    A[MIMIC-IV / EHR Data] --> B[Clinical Preprocessing]
    B --> C[Patient Timelines]
    C --> D[Health Trajectory Engine]
    D --> E[Trajectory & State Features]
    E --> F[Offline RL: Discrete CQL]
    F --> G[Monitoring Recommendation]
    G --> H[Explainable AI: SHAP]
    H --> I[Cumulative Health Report]
    I --> J[Healthcare Dashboard & REST API]
```

### Complete Pipeline Stages
1. **Historical & Current Healthcare Data**: MIMIC-IV 3.1 hospital database (diagnoses, labs, OMR vitals, demographics) or calibrated longitudinal cohort.
2. **Clinical Preprocessing**: Date standardization, duplicate removal, physiological bounds filtering, patient-level forward/backward fill + population median imputation fallback, categorical encoding.
3. **Patient Timeline**: Chronological ordering, sequential visit numbering (1, 2, 3...), and elapsed interval calculation ($\Delta t$ in days).
4. **Health Trajectory Analysis Engine**: Calculates absolute biomarker changes ($\Delta x$), monthly rates of change ($(\Delta x / \Delta t) \times 30$), moving averages, and clinical trend directions on **raw clinical values prior to normalization** to preserve clinical interpretability.
5. **Leak-Free Normalization**: `StandardScaler` fitted strictly on training cohort patients and applied to numerical and continuous trajectory features.
6. **Patient State Formulation**: Normalized state vector $s_t \in \mathbb{R}^D$ combining clinical indicators, trajectory features, and patient context.
7. **Offline Reinforcement Learning (Discrete CQL)**: Evaluates the evolving state and recommends a proactive monitoring or risk-management action (Routine Monitoring, Increased Monitoring, Early-Risk Alert, Clinical Evaluation) without prescribing medication.
8. **Explainable AI (SHAP)**: Identifies the most influential clinical indicators and trajectory rates driving the selected action's Q-value, translating them into human-readable clinical narratives.
9. **Cumulative Health Report**: Assembles the patient's demographics, trajectory history, RL decision, and XAI rationale into JSON, Markdown, and an interactive HTML dashboard.
10. **FastAPI REST API**: Production-ready endpoints for patient queries, trajectory analysis, real-time action predictions, and report generation.

---

## 2. Mathematical Formulations

### Health Trajectory Analysis
For biomarker $x$ observed at encounter dates $t_i$ and $t_{i-1}$ with elapsed days $\Delta t = t_i - t_{i-1}$:
- **Absolute Change ($\Delta x$)**:
  $$\Delta x_i = x_i - x_{i-1}$$
- **Monthly Rate of Change**:
  $$\text{Rate}_i = \left( \frac{x_i - x_{i-1}}{\Delta t} \right) \times 30.0$$
- **Rolling Moving Average**:
  $$\text{MA}_i = \frac{1}{K} \sum_{k=0}^{K-1} x_{i-k} \quad (K=3)$$
- **Clinical Trend Direction**:
  $$\text{Trend}(x_i) = \begin{cases} +1 \text{ (increasing)}, & \text{if } \Delta x_i \ge \theta_{\text{sig}} \\ -1 \text{ (decreasing)}, & \text{if } \Delta x_i \le -\theta_{\text{sig}} \\ 0 \text{ (stable)}, & \text{otherwise} \end{cases}$$
  *(e.g., for HbA1c, ADA guideline threshold $\theta_{\text{sig}} = 0.3\%$)*

### Discrete Conservative Q-Learning (CQL)
Standard DQN algorithms suffer from dangerous overestimation bias when trained on offline observational EHR data because the maximization operator queries out-of-distribution (OOD) actions.

Discrete CQL penalizes the Q-values of out-of-distribution actions while maximizing the expected Q-value under the data distribution:
$$\mathcal{L}_{\text{CQL}}(\theta) = \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_{a \in \mathcal{A}} \exp(Q(s, a)) - Q(s, a_{\text{data}}) \right]$$

Combined with Double Q-learning Bellman Temporal Difference (TD) loss:
$$\mathcal{L}_{\text{TD}}(\theta) = \frac{1}{2} \mathbb{E}_{(s, a, r, s')} \left[ \left( Q(s, a) - y \right)^2 \right]$$
$$y = r + \gamma (1 - d) \min_{j=1,2} \hat{Q}_j\left(s', \arg\max_{a'} Q_1(s', a')\right)$$
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{TD}}(\theta) + \alpha \cdot \mathcal{L}_{\text{CQL}}(\theta)$$

### Clinically Grounded Reward Function
The reward $R(s_t, a_t, s_{t+1})$ focuses on monitoring timeliness and early risk detection:
- **Deteriorating Trajectory**:
  - `EARLY_RISK_ALERT` ($a=2$): $+3.5$ (Proactive early warning)
  - `INCREASED_MONITORING` ($a=1$): $+2.0$ (Appropriate closer follow-up)
  - `ROUTINE_MONITORING` ($a=0$): $-3.5$ (Under-monitoring / clinical negligence)
- **Rapidly Deteriorating / Critical Trajectory**:
  - `CLINICAL_EVALUATION` ($a=3$): $+3.0$ (Urgent specialist intervention)
  - `ROUTINE_MONITORING` ($a=0$): $-4.0$ (Critical negligence)
- **Stable / Well-Controlled Trajectory**:
  - `ROUTINE_MONITORING` ($a=0$): $+2.5$ (Resource efficiency, zero alert fatigue)
  - `EARLY_RISK_ALERT` / `CLINICAL_EVALUATION`: $-2.5$ to $-3.5$ (Alarm fatigue penalty)

---

## 3. Directory Structure

```
diabetes-cdss/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep                   # Directory placeholder for raw MIMIC-IV 3.1 extracts
│   ├── interim/
│   │   └── t2d_cohort_raw.csv         # Staged raw longitudinal T2D cohort records (10 biomarkers)
│   ├── processed/
│   │   ├── trajectory_data.csv        # 90-feature longitudinal dataset (deltas, rates, moving avgs, trends)
│   │   ├── rl_dataset.csv             # Offline RL transitions log (s, a, r, s', done)
│   │   └── preprocessing_objects/
│   │       ├── scaler.pkl             # Fitted ClinicalScaler (StandardScaler on train split)
│   │       ├── encoders.pkl           # Fitted categorical encoders (gender, smoking, activity)
│   │       └── imputer.pkl            # Longitudinal baseline medians & categorical modes
│   └── feedback/
│       ├── clinician_feedback.csv     # Clinician review audit log, agreements & expert overrides
│       ├── pending_recommendations.csv# Active recommendations awaiting future patient follow-ups
│       └── completed_transitions.csv  # Resolved (s, a, r, s', done) feedback tuples for continual fine-tuning
│
├── preprocessing/
│   ├── __init__.py
│   ├── load_data.py                   # MIMIC-IV chunked extractor & calibrated cohort generator
│   ├── cleaning.py                    # Date parsing, deduplication, sanity bounds
│   ├── missing_values.py              # Patient-level ffill/bfill & baseline median fallback
│   ├── feature_engineering.py         # Visit index, delta-time (dt), cumulative days
│   ├── encoding.py                    # Categorical encoders (gender, smoking, activity)
│   ├── normalization.py               # Leak-free StandardScaler (fitted on train split only)
│   └── pipeline.py                    # Full preprocessing pipeline orchestrator & artifact persistence
│
├── trajectory/
│   ├── __init__.py
│   ├── timeline.py                    # Patient timeline construction and dt calculation
│   ├── changes.py                     # Absolute deltas (x_t - x_{t-1}) and monthly rates
│   ├── trends.py                      # Clinical trend direction classification
│   ├── moving_average.py              # 3-visit rolling MA and exponential smoothing (EMA)
│   ├── trajectory_features.py         # Trajectory representation assembler
│   └── pipeline.py                    # Trajectory pipeline & overall status classification
│
├── rl/
│   ├── __init__.py
│   ├── state.py                       # Normalized state vector constructor (s_t in R^D)
│   ├── actions.py                     # Discrete monitoring actions (0 to 3) & metadata
│   ├── reward.py                      # Clinically grounded monitoring reward function
│   ├── transitions.py                 # Offline transition generation (s, a, r, s', done)
│   ├── dataset.py                     # PyTorch Dataset & patient-partitioned DataLoaders
│   ├── model.py                       # Discrete CQL Twin Q-Network architecture
│   ├── cql.py                         # CQL training algorithm and loss formulation
│   ├── feedback.py                    # Clinician RLHF manager & continual experience replay buffer
│   ├── train.py                       # Discrete CQL training loop with conservative loss
│   └── evaluate.py                    # Policy evaluation vs clinical rule/behavior baselines
│
├── xai/
│   ├── __init__.py
│   ├── shap_explainer.py              # SHAP explainer for CQL Q-network
│   ├── feature_importance.py          # Top factor ranking and clinical domain grouping
│   └── explanation_generator.py       # Human-readable natural language clinical rationale
│
├── report/
│   ├── __init__.py
│   ├── cumulative_report.py           # Cumulative report dataclass schemas
│   └── report_generator.py            # JSON, Markdown, and interactive HTML dashboard
│
├── api/
│   ├── __init__.py
│   ├── app.py                         # Consolidated FastAPI application, CORS & lifespan
│   ├── main.py                        # FastAPI entry point
│   ├── patient_routes.py              # Patient listing & chronological timeline endpoints
│   ├── prediction_routes.py           # Action prediction & SHAP explanation endpoints
│   └── report_routes.py               # Cumulative health report & HTML rendering endpoints
│
├── models/
│   ├── rl_agent/                      # Trained CQL model checkpoint (cql_model.pt)
│   ├── scalers/                       # ClinicalScaler artifact
│   └── encoders/                      # CategoricalEncoder artifact
│
├── static/
│   └── index.html                     # Clinician web dashboard UI (visualizer, XAI, feedback)
│
├── reports/                           # Generated cumulative patient reports (JSON, MD, HTML)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Cohort EDA, biomarker distributions, follow-up gaps
│   ├── 02_preprocessing.ipynb         # Cleaning, imputation, and leakage-free splitting
│   ├── 03_trajectory_analysis.ipynb   # Trajectory feature calculation and visualization
│   ├── 04_rl_training.ipynb           # Offline RL dataset construction & CQL training
│   └── 05_evaluation.ipynb            # Policy evaluation, SHAP attributions, cumulative reports
│
├── tests/
│   ├── test_preprocessing.py          # Unit tests for cleaning, imputation, and scaling
│   ├── test_trajectory.py             # Unit tests for timeline, deltas, rates, trends
│   ├── test_rl.py                     # Unit tests for state, actions, rewards, and CQL network
│   ├── test_feedback_retraining.py    # Unit tests for clinician feedback and continual fine-tuning
│   └── test_api.py                    # Integration tests for FastAPI endpoints
│
├── test_model_accuracy.py             # Comprehensive model accuracy, confusion matrix & baseline bench
├── config.py                          # Global configuration (paths, clinical thresholds, RL hypers)
├── requirements.txt                   # Required Python dependencies
├── main.py                            # Unified CLI pipeline runner
└── README.md                          # System documentation
```

### Data Folder Architecture & Schema Reference

The `data/` directory is organized into four progressive tiers following clinical machine learning best practices:

| Subfolder | File Name | Format | Primary Role & Description |
| :--- | :--- | :--- | :--- |
| **`raw/`** | [`.gitkeep`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/raw/.gitkeep) | Text | Directory placeholder for raw MIMIC-IV 3.1 hospital database extracts (`hosp/labevents`, `hosp/diagnoses_icd`, `hosp/omr`, `hosp/patients`). Not tracked in Git per PhysioNet DUA compliance. |
| **`interim/`** | [`t2d_cohort_raw.csv`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/interim/t2d_cohort_raw.csv) | CSV | Staged raw longitudinal T2D cohort records. Contains 1,205 encounter rows across patients with un-imputed missing values, encounter dates, demographics, and 10 clinical biomarkers (`hba1c`, `glucose`, `bmi`, `sbp`, `dbp`, `cholesterol`, `ldl`, `hdl`, `triglycerides`, `creatinine`). |
| **`processed/`** | [`trajectory_data.csv`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/processed/trajectory_data.csv) | CSV | **Core longitudinal trajectory dataset** (1,215 encounters, 90 columns). Combines cleaned and imputed biomarkers with chronological features (`visit_number`, `days_since_last_visit`, `cumulative_days`), biomarker dynamics (absolute changes $\Delta x$, monthly velocity rates $(\Delta x / \Delta t) \times 30$, 3-visit rolling moving averages, exponential moving averages), ADA guideline-calibrated trend classifications (`stable`, `increasing`, `decreasing`), and overall clinical status (`stable`, `deteriorating`, `rapidly_deteriorating`, `improving`). |
| **`processed/`** | [`rl_dataset.csv`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/processed/rl_dataset.csv) | CSV | Offline Reinforcement Learning transitions log (1,214 steps). Maps sequential patient encounters into Markov Decision Process (MDP) tuples: `subject_id`, `visit_number`, `action` (0–3), `reward` (clinically grounded reward), `done` (terminal flag), and trajectory status. Directly used to train and evaluate the Discrete CQL policy. |
| **`processed/preprocessing_objects/`** | [`scaler.pkl`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/processed/preprocessing_objects/scaler.pkl) | Pickle | Fitted `ClinicalScaler` (`StandardScaler`) artifact fitted strictly on training cohort patients. Normalizes continuous biomarkers and trajectory features with zero data leakage. Loaded during inference to scale new patient visits. |
| **`processed/preprocessing_objects/`** | [`encoders.pkl`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/processed/preprocessing_objects/encoders.pkl) | Pickle | Fitted `CategoricalEncoder` dictionary mapping categorical variables (`gender`, `smoking_status`, `physical_activity`) to integer indices. |
| **`processed/preprocessing_objects/`** | [`imputer.pkl`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/processed/preprocessing_objects/imputer.pkl) | Pickle | Fitted `LongitudinalImputer` state storing baseline training population medians and categorical modes used as fallback for patient encounters with unobserved lab tests. |
| **`feedback/`** | [`clinician_feedback.csv`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/feedback/clinician_feedback.csv) | CSV | Audit log for physician interactions in the dashboard/API: logs `feedback_id`, `timestamp`, `subject_id`, `model_action`, `clinician_action`, `is_override` (0 or 1), physician clinical reasoning `notes`, and encoded state vectors. |
| **`feedback/`** | [`pending_recommendations.csv`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/feedback/pending_recommendations.csv) | CSV | Tracks active recommendations awaiting the patient's future follow-up encounter to resolve delayed biological rewards. When the patient returns for their next visit, `register_patient_encounter()` retrieves the pending record, computes the true biological outcome reward, marks status as `fulfilled`, and converts it into a completed transition. |
| **`feedback/`** | [`completed_transitions.csv`](file:///f:/FYDP/XRL%20based%20CDSS/diabetes-cdss/data/feedback/completed_transitions.csv) | CSV | Closed $(s, a, r, s', done)$ transition tuples generated from clinician expert overrides and fulfilled longitudinal follow-ups. Merged via Experience Replay (`export_retraining_dataset()`) for safe continual policy fine-tuning without catastrophic forgetting. |

---

## 4. Installation & Setup

```bash
# Clone or navigate to the project directory
cd diabetes-cdss

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Usage & Execution Modes

The unified CLI `main.py` runs all pipeline stages:

### Run Complete End-to-End Demo
Executes extraction, preprocessing, trajectory analysis, leak-free normalization, Discrete CQL training, benchmark evaluation, SHAP explainability, and report generation in a single command:
```bash
python main.py --mode demo
```

### Run Individual Pipeline Stages
```bash
# 1. Extract cohort from MIMIC-IV or generate synthetic cohort
python main.py --mode extract --source auto --num-patients 500

# 2. Preprocess raw data and fit artifacts
python main.py --mode preprocess

# 3. Compute longitudinal trajectory features and trend directions
python main.py --mode trajectory

# 4. Train Discrete CQL Offline RL agent
python main.py --mode train --epochs 30

# 5. Evaluate trained policy against clinical heuristics and behavior baselines
python main.py --mode evaluate

# 6. Test Training vs Testing Accuracy, Generalization Gap & Confusion Matrix
python test_model_accuracy.py
# or:
python main.py --mode accuracy

# 7. Generate cumulative report for a specific patient
python main.py --mode report --patient-id P1001

# 8. Launch FastAPI REST API server
python main.py --mode serve --port 8000
```

---

## 6. REST API Documentation

When running `python main.py --mode serve`, the interactive Swagger UI is available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API health check |
| `GET` | `/api/patients` | List all registered patients with summary metadata |
| `GET` | `/api/patients/{subject_id}/timeline` | Retrieve chronological trajectory history |
| `GET` | `/api/patients/{subject_id}/latest` | Retrieve latest observation and trajectory status |
| `POST` | `/api/predict/action` | Predict optimal RL monitoring action and Q-values |
| `POST` | `/api/predict/explain` | Compute SHAP attributions and natural language rationale |
| `GET` | `/api/reports/{subject_id}` | Retrieve cumulative health report in JSON format |
| `GET` | `/api/reports/{subject_id}/markdown` | Retrieve cumulative health report in Markdown |
| `GET` | `/api/reports/{subject_id}/html` | Render interactive HTML dashboard in browser |

---

## 7. Running Unit Tests

Run the complete test suite with `pytest`:
```bash
pytest tests/ -v
```
All unit tests verify:
- Preprocessing and train/test patient separation (zero leakage)
- Timeline ordering, biomarker $\Delta x$, rates of change, moving averages, and clinical trend categorizations
- RL state representation, 4-class monitoring actions, clinical reward boundaries, and CQL network tensor shapes.

---

## 8. Data Privacy & MIMIC-IV Reproducibility

### Strict Privacy Compliance (PhysioNet DUA)
In accordance with the **PhysioNet Credentialed Health Data Use Agreement (DUA)** and HIPAA regulations:
- No raw, de-identified, or transformed MIMIC-IV hospital data records (`mimic-iv-3.1/`, `data/**/*.csv`) are stored or tracked in this Git repository.
- `.gitignore` automatically excludes all clinical datasets and individual patient-level reports.

### Instant Reproduction (No MIMIC-IV Required)
Anyone who clones this repository can immediately run the entire pipeline without needing PhysioNet credentials. The system automatically falls back to an embedded, clinically calibrated synthetic generator:
```bash
# Automatically generates synthetic longitudinal cohort, trains CQL agent, and evaluates policy
python main.py --mode demo
```

### For Credentialed PhysioNet Researchers
If you have authorized access to MIMIC-IV:
1. Download `mimic-iv-3.1` (specifically `hosp/diagnoses_icd.csv.gz`, `hosp/patients.csv.gz`, `hosp/labevents.csv.gz`, `hosp/omr.csv.gz`).
2. Place the folder at the root as `mimic-iv-3.1/hosp` or configure the path in `config.py`:
   ```python
   MIMIC_IV_ROOT = Path("/path/to/your/mimic-iv-3.1/hosp")
   ```
3. Run extraction and preprocessing:
   ```bash
   python main.py --mode preprocess --source mimic
   ```

